
from typing import Optional
import numpy as np
import gymnasium as gym
import pandas as pd
import torch
from tqdm import tqdm
from utils.data_prep_for2D import load_data
from scipy.special import expit

"""
Version 5: with 2D data generated
Similarly, use the data processed by methods from "Performative Prediction"
"""

# hyperparameters
test_label_threshold = 0.5  # threshold for the test label
seed = 0 # 0 or 2
# cost parameter: v_i = 0.5 -> epsilon = 1 (different form in different papers)
epsilon: float = 0.5 # larger: easier to manipulate, smaller: harder to manipulate
strat_features = np.array([0, 1])
strategic_response = True
response_method = "Close"  # "GA" or "Close"
dimension = 2 # number of features

class creditScoring_v5(gym.Env):

    def __init__(self,
                 policy_weight=[0.1]*(dimension + 1),  # +1 for bias
                 maximum_episode_length: int = 1000000):
        
        self.mode = 'train'
        self.maximum_episode_length = maximum_episode_length

        # obsevation space: 11-dimensional vector (10 features + 1 bias term)
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            # high = 10000000,
            shape=(dimension+1,),
            dtype=np.float64
        )

        # action space: discrete action space with 2 actions (0 or 1)
        self.action_space = gym.spaces.Discrete(2)

        # Define the pointer to the sample for online learning
        self.samplePointer = 0

        # Load the training and test data
        filePath = "./data/generated_2D_data.csv"
        self.train_x, self.train_y, self.test_x, self.test_y = load_data(filePath, seed=seed)

        # parameter of the real cost function
        # Assume using a weighted quadratic cost function (same as in the "made practical" paper)
        self.cost_weight = np.full(shape=10, fill_value=0.5, dtype=np.float64)
        
        # 把 policy_weight 从 list 转成 ndarray
        self.policy_weight = np.asarray(policy_weight, dtype=np.float64)

        # test
        self.trigger_once = False
    
    def strategic_response_GA(self,
                          real_feature: np.ndarray,
                          policy_weight: np.ndarray,
                          learning_rate: float = 0.01,
                          num_steps: int = 20,
                          epsilon: float = epsilon,
                          strat_features: Optional[list] = None):
        """
        Strategic response using gradient ascent to maximize f(z) - cost, updating only strat_features.
        Utility: f(z) = sigmoid(theta^T z_full); cost = 1/(2*epsilon) * |z_s - x_s|^2.
        Only strat_features are optimized; other features and bias stay fixed.
        """
        # 初始化调用计数
        if not hasattr(self, "_response_call_count"):
            self._response_call_count = 0
        self._response_call_count += 1

        # 如果不开启战略响应，直接返回原特征
        if not strategic_response:
            return real_feature

        # 默认操纵所有非 bias 特征
        n_features = len(policy_weight) - 1
        if strat_features is None:
            strat_features = list(range(n_features))
        # non-strategic 特征索引
        ns_features = [i for i in range(n_features) if i not in strat_features]

        # 选择设备并转换数据类型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_orig = torch.tensor(real_feature.astype(np.float32), device=device)
        theta_full = torch.tensor(policy_weight.astype(np.float32), device=device)
        # 分离权重和 bias
        theta = theta_full[:-1]    # 特征权重
        bias = theta_full[-1]      # 偏置权重

        # strategic / non-strategic 部分
        x_s  = x_orig[strat_features]
        x_ns = x_orig[ns_features]
        # remove bias from x_ns
        x_ns = x_ns[:-1]  # bias 是最后一个元素
        theta_s  = theta[strat_features]
        theta_ns = theta[ns_features]

        # 成本系数
        cost_s = torch.tensor(self.cost_weight[strat_features].astype(np.float32), device=device)

        # 预计算非-strategic 与 bias 的常量项
        with torch.no_grad():
            const_term = torch.dot(theta_ns, x_ns) + bias

        # 需要优化的变量
        z_s = x_s.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z_s], lr=learning_rate)

        # 是否记录轨迹
        record = self._response_call_count in {10, 20, 30, 40, 50}
        history = []

        # 梯度上升迭代
        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = torch.dot(theta_s, z_s) + const_term
            fz = torch.sigmoid(logits)
            # 成本项带 epsilon 调节
            cost = torch.sum(cost_s * (z_s - x_s) ** 2) / (2 * epsilon)
            loss = -fz + cost
            loss.backward()
            optimizer.step()

            # 投影到 [-5,5]
            with torch.no_grad():
                z_s.clamp_(-5.0, 5.0)

            if record:
                history.append(z_s.detach().cpu().numpy().copy())

        # 可选可视化
        if record and history:
            import matplotlib.pyplot as plt
            arr = np.stack(history, axis=0)
            plt.figure(figsize=(10, 5))
            for i, f in enumerate(strat_features):
                plt.plot(arr[:, i], label=f'z[{f}]')
            plt.title(f"z Convergence (call #{self._response_call_count})")
            plt.xlabel('Step')
            plt.ylabel('z value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"./result/last_experiment/z_conv_call_{self._response_call_count}.png")
            plt.close()

        # 合成最终特征向量
        modified = real_feature.copy()
        modified[strat_features] = z_s.detach().cpu().numpy()

        modified = np.clip(modified, -5.0, 5.0)
        return modified

    def strategic_response_Close_naive(self, 
                        real_feature: np.ndarray, 
                        policy_weight: np.ndarray,
                        epsilon: float = epsilon,  # manipulation strength
                        strat_features: Optional[list] = strat_features):
        """
        Best response function with linear utility and quadratic cost.
        Only manipulates features in strat_features.
        Moves only when misclassified & not too far from the boundary (like optimize_X).
        """

        if not strategic_response:
            return real_feature

        if strat_features is None:
            strat_features = list(range(len(policy_weight) - 1))  # exclude bias term

        # 1. 分离权重和偏置
        theta = policy_weight[:-1]  # classifier weight
        bias = policy_weight[-1]    # classifier bias

        # 2. 计算 margin = theta^T x + b
        margin = np.dot(policy_weight, real_feature)

        # 3. 模拟 optimize_X 中的判断逻辑
        # 假设用户希望预测为正类 → r = 1
        # 如果你需要 r = -1 的情况，请根据调用时修改
        r = -1.0
        directional_margin = r * margin
        if directional_margin >= 0 or directional_margin < -2:
            # 不需要操控（已经正确 or 距离太远）
            return real_feature

        # 4. 只修改 strategy features
        modified = np.copy(real_feature)
        theta_strat = theta[strat_features]
        modified[strat_features] += -epsilon * theta_strat

        # 5. 限制范围
        modified = np.clip(modified, -10.0, 10.0)

        return modified

    def strategic_response_Close(self, 
                    real_feature: np.ndarray, 
                    policy_weight: np.ndarray,
                    safety_overshoot: float = 1e-2,  # small small safety overshoot
                    strat_features: Optional[list] = None,
                    strategic_response: bool = True,
                    r: float = -1.0):  # 用户希望被预测为 0 → r = -1
        """
        Cost-sensitive best response:
        Only manipulates features in strat_features,
        and only moves if (1) misclassified and (2) cost is acceptable (≤2).

        Logic mimics 'optimize_X'.
        """
        if not strategic_response:
            return real_feature

        if strat_features is None:
            strat_features = list(range(len(policy_weight) - 1))  # exclude bias

        # 1. 分离权重和偏置
        theta = policy_weight[:-1]  # classifier weight
        bias = policy_weight[-1]    # classifier bias

        # 2. 当前点的 margin（带偏置）: theta^T x + b
        margin = np.dot(theta, real_feature[:-1]) + bias

        # 3. 归一化方向 & 距离
        theta_norm = np.linalg.norm(theta)
        directional_margin = r * margin
        signed_dist = directional_margin / theta_norm  # signed distance

        # 4. 判断是否需要操控
        # 4.1 如果已经预测正确 or 距离太远，则不动
        if signed_dist >= 0:
            return real_feature
        
        # 4.2 模拟将移动的点（按 epsilon 更新）
        # 显式计算 cost-benefit trade-off
        # 如果带来收益小于代价，则不动

        # 沿 theta 的方向移动，直到靠近边界（略过一点）
        move_direction = -(1 + safety_overshoot) * r * signed_dist * theta / theta_norm
        modified = np.copy(real_feature)
        # 仅对 strat_features 应用 move_direction 中的对应分量
        for i in strat_features:
            modified[i] += move_direction[i]
        benefit = 1.0  # hyperparameter
        cost = np.sum((modified - real_feature)**2) / (2 * epsilon)
        if cost > benefit:
            return real_feature
        # print(f"Moving from {real_feature} to {modified} with cost {cost} and benefit {benefit}")
        
        # 5. 限制范围
        modified = np.clip(modified, -5.0, 5.0)

        return modified


    # def load_test_data(self):
    #     path = "data/ProcessedData/"

    #     test_data = pd.read_csv(path + "cs-test-processed.csv")
    #     test_prob = pd.read_csv(path + "sampleEntry.csv")
    #     # test_data["NumberOfDependents"] = test_data["NumberOfDependents"].astype(int)
    #     # test_data["MonthlyIncome"] = test_data["MonthlyIncome"].astype(int)

    #     # extract the target column and the features
    #     test_x = test_data.drop(columns=['SeriousDlqin2yrs']).to_numpy()
    #     test_x = np.append(test_x, np.ones((test_x.shape[0], 1)), axis=1)
    #     test_y = test_prob.drop(columns=['Id']).to_numpy()  # drop the ID column
    #     test_y = test_y.ravel() # flatten the target to 1D array
    #     test_y = (test_y >= test_label_threshold).astype(int)
        
    #     return test_x, test_y
    
    # called in .reset() & .step()

    def _get_obs(self):
        if self.mode == 'train':
            sample = self.train_x[self.samplePointer]
        else:
            sample = self.test_x[self.samplePointer]
        
        # response to the sample
        if response_method == "GA":
            observation = self.strategic_response_GA(sample, self.policy_weight)
        elif response_method == "Close":
            observation = self.strategic_response_Close(sample, self.policy_weight)
        else:
            raise ValueError(f"Unknown response method: {response_method}")
        
        # log
        if not self.trigger_once:
                print(f"sample: {sample}, modified: {observation}")
                self.trigger_once = True
        
        return observation
    
    def _get_info(self):
        # 当前样本的真值
        if self.mode == 'train':
            target = self.train_y[self.samplePointer]
            max_len = len(self.train_x)
        else:
            target = self.test_y[self.samplePointer]
            max_len = len(self.test_x)

        # 计算下一个指针，检查是否越界
        next_idx = self.samplePointer + 1
        if next_idx < max_len:
            # 合法时再生成 next_obs，否则直接 None
            sample_seq = self.train_x if self.mode == 'train' else self.test_x
            next_sample = sample_seq[next_idx]

            if response_method == "GA":
                next_obs = self.strategic_response_GA(next_sample, self.policy_weight)
            elif response_method == "Close":
                next_obs = self.strategic_response_Close(next_sample, self.policy_weight)
            else:
                # non-strategic 下直接 None
                next_obs = None
        else:
            next_obs = None

        return {
            'true_label': target,
            'next_obs': next_obs
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the sample pointer to the beginning of the training data
        self.samplePointer = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action, previous_policy_weight=None):
        """
        Execute one step in the environment for the current sample, returning the next observation,
        the reward according to the original mechanism, termination flags, and the true label of the current sample.

        Returns:
            next_obs: Observation for the next sample (or None if terminated/truncated).
            reward: +1 for correct (action=0,label=0), +0.5 for (action=1,label=1), -0.5 for (action=1,label=0), -1 for (action=0,label=1).
            terminated: True if no more samples remain.
            truncated: True if sample count exceeds maximum_episode_length.
            info: Dict containing the true label of the current sample.
        """
        # 1) Compute reward using the original credit scoring logic (samplePointer remains unchanged)
        y_seq = self.train_y if self.mode == 'train' else self.test_y
        label = y_seq[self.samplePointer]
        if action == 0 and label == 0:
            reward = +1
        elif action == 0 and label == 1:
            reward = -1
        elif action == 1 and label == 0:
            reward = -0.5
        else:  # action == 1 and label == 1
            reward = +0.5

        # 2) Update policy weight if provided (optional for strategic responses)
        if previous_policy_weight is not None:
            self.policy_weight = previous_policy_weight

        # 3) Prepare next index and check termination/truncation
        seq_x = self.train_x if self.mode == 'train' else self.test_x
        max_len = len(seq_x)
        next_idx = self.samplePointer + 1
        terminated = next_idx >= max_len
        truncated = next_idx > self.maximum_episode_length

        # 4) Record the true label of the current sample
        info = {'true_label': label}

        # 5) Advance pointer and fetch next observation if not terminated or truncated
        if not (terminated or truncated):
            self.samplePointer = next_idx
            next_obs = self._get_obs()
        else:
            next_obs = None

        return next_obs, reward, terminated, truncated, info


# Register the environment after the class definition
gym.register(
    id="creditScoring_v5",
    entry_point="env.creditScoring_v5:creditScoring_v5"
)

if __name__ == "__main__":
    env = creditScoring_v5()


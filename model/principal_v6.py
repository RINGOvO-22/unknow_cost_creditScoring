from collections import deque
import gymnasium as gym
import numpy as np
from scipy.special import expit

"""
Version 6: cost-aware SRL (Fully Fixed)
- observation: [x1, ..., x10, bias=1.0] → dim=11
- 只对前 10 维特征进行 cost estimation 和 manipulability 计算
- 使用 MAD 动态估计每维特征的操纵难易度
- 在策略梯度中加入 manipulability 惩罚项
"""

# default hyperparameters
default_learning_rate = 1e-3
default_discount_factor = 0.99
default_predict_label_threshold = 0.5
default_batch_size = 128
default_init_cost_pram = 2.0

# clipping
clipVal_td = 10.0
clipVal_grad_log_pi = 10.0
clipVal_policyWeight = 10.0

# random seed
np.random.seed(0)


class Principal_v6:
    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 10,  # 显式传入真实特征维度（不包括 bias）
        learning_rate_critic: float = default_learning_rate,
        learning_rate_actor: float = default_learning_rate,
        learning_rate_cost: float = 0.1,  # cost belief 更新率
        buffer_size: int = default_batch_size,
        discount_factor: float = 0.99,
        init_cost_pram: float = 2.0,  # 初始 cost 估计
        lambda_penalty: float = 0.1,  # manipulability 惩罚系数
        epsilon: float = 1e-6,
        costAware_flag = True  # 是否使用 cost-aware 策略
    ):
        self.env = env
        self.feature_dim = feature_dim
        self.discount_factor = discount_factor
        self.lr_a = learning_rate_actor
        self.lr_c = learning_rate_critic
        self.lr_cost = learning_rate_cost
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.costAware_flag = costAware_flag

        # policy weight: [w1, ..., w10, b] → 长度 11
        self.previous_policy_weight = np.random.normal(loc=0.0, scale=0.1, size=(feature_dim + 1,))
        # Q 网络权重: [w1,...,w10,b,a] → 输入是 [obs (11), action (1)] → 长度 12
        self.q_weights = np.ones(feature_dim + 1 + 1, dtype=np.float64) * 0.01  # 11 for obs + 1 for action

        # cost_belief: 每个特征维度一个值，不包括 bias → 长度 10
        self.cost_belief = np.full(shape=feature_dim, fill_value=init_cost_pram, dtype=np.float64)

        # replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # logging
        self.batch_update_count = 0
        self.training_batch_acc = []
        self.training_expected_acc_list = []
        self.training_rewards = []
        self.training_policy_weights = []
        self.training_single_policy_weight_update = []
        self.training_manipulability = []  # 新增：记录 manipulability
        self.training_acc_detail = []
        self.testing_accuracy = []
        self.testing_acc_detail = []

    def get_action(self, obs: np.ndarray, stochastic=True) -> tuple:
        """
        obs: shape (11,) = [x1,...,x10, bias=1.0]
        """
        logits = np.dot(self.previous_policy_weight, obs)  # 自动包含 bias
        prob = expit(logits)
        action = np.random.binomial(n=1, p=prob) if stochastic else int(prob > default_predict_label_threshold)
        return prob, action

    def compute_manipulability(self, batch_obs: np.ndarray) -> float:
        """
        batch_obs: shape (B, 11), 最后一维是 bias
        只对前 10 维（可操纵特征）进行计算
        """
        B, D = batch_obs.shape
        if D != self.feature_dim + 1:
            raise ValueError(f"Expected observation dim {self.feature_dim + 1}, but got {D}")

        features = batch_obs[:, :self.feature_dim]  # shape: (B, 10)
        d = self.feature_dim
        sensitivity = np.zeros(d)

        # 1. 计算每个特征维度的敏感度 |∂π/∂z_i|
        # π = σ(w^T z + b)
        # ∂π/∂z_i = σ'(logits) * w_i
        logits = np.dot(features, self.previous_policy_weight[:d]) + self.previous_policy_weight[d]
        grad_sigma = expit(logits) * (1 - expit(logits))  # shape: (B,)

        for i in range(d):
            w_i = self.previous_policy_weight[i]
            sensitivity[i] = np.mean(np.abs(grad_sigma * w_i))

        # 2. 使用 MAD 计算“易操纵性” (1 / MAD)
        ease_of_manip = np.zeros(d)
        for i in range(d):
            mad = np.mean(np.abs(features[:, i] - np.median(features[:, i])))
            ease_of_manip[i] = 1 / (mad + self.epsilon)

        # 3. 更新 cost_belief: 滑动平均
        self.cost_belief = (1 - self.lr_cost) * self.cost_belief + self.lr_cost * ease_of_manip

        # 4. manipulability = sum_i sensitivity_i * ease_of_manip_i
        # 注意：ease_of_manip 已是 1/MAD，代表“易被操纵”，所以值越大越危险
        manipulability = np.sum(sensitivity * self.cost_belief)

        return manipulability

    def batch_update(self):
        if len(self.buffer) < self.buffer_size:
            return

        self.batch_update_count += 1
        batch = list(self.buffer)
        np.random.shuffle(batch)

        # 提取 batch 中的 obs（用于 cost estimation）
        batch_obs = np.array([sample[0] for sample in batch])  # shape: (B, 11)

        # === Step 1: Compute manipulability ===
        manipulability = self.compute_manipulability(batch_obs)
        self.training_manipulability.append(manipulability)

        # === Step 2: Critic & Actor 更新 ===
        accs = []

        for obs, action, reward, terminated, next_obs, true_label in batch:
            # --- Critic Update ---
            q_input = np.append(obs, action)  # shape: 11 + 1 = 12
            q_value = np.dot(self.q_weights, q_input)

            if not terminated and next_obs is not None:
                q_next_0 = np.dot(self.q_weights, np.append(next_obs, 0))
                q_next_1 = np.dot(self.q_weights, np.append(next_obs, 1))
                max_q_next = max(q_next_0, q_next_1)
            else:
                max_q_next = 0.0

            td_target = reward + self.discount_factor * max_q_next
            td_error = td_target - q_value
            td_error = np.clip(td_error, -clipVal_td, clipVal_td)

            self.q_weights += self.lr_c * td_error * q_input

            # --- Actor Update ---
            logits = np.dot(self.previous_policy_weight, obs)
            prob = expit(logits)
            grad_log_pi = (action - prob) * obs  # shape: (11,)
            grad_log_pi = np.clip(grad_log_pi, -clipVal_grad_log_pi, clipVal_grad_log_pi)

            # 使用 cost-aware advantage
            if self.costAware_flag:
                advantage = td_error - self.lambda_penalty * manipulability
            else:
                advantage = td_error

            weight_update = self.lr_a * advantage * grad_log_pi / self.buffer_size
            weight_update = np.clip(weight_update, -clipVal_policyWeight, clipVal_policyWeight)

            self.previous_policy_weight += weight_update
            self.training_single_policy_weight_update.append(weight_update)

            # 记录 expected accuracy
            correct_prob = prob if true_label == 1 else (1 - prob)
            accs.append(correct_prob)

        self.buffer.clear()
        self.training_batch_acc.append(np.mean(accs))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        info: dict,
        prob: float
    ):
        true_label = info['true_label']
        next_obs = info.get('next_obs', None)
        sample = (obs, action, reward, terminated, next_obs, true_label)
        self.buffer.append(sample)

        if len(self.buffer) >= self.buffer_size:
            self.batch_update()

        # 记录单步
        pred = action
        correct_prob = prob if true_label == 1 else (1 - prob)
        self.training_expected_acc_list.append(correct_prob)
        self.training_rewards.append(reward)
        self.training_policy_weights.append(self.previous_policy_weight.copy())
        self.training_acc_detail.append({
            'predicted_prob': prob,
            'predicted_label': pred,
            'true_label': true_label,
            'expected_accuracy': correct_prob,
            'reward': reward
        })

    def test_result_record(self, action: int, info: dict, prob: float):
        accuracy = 1.0 if action == int(info['true_label']) else 0.0
        self.testing_accuracy.append(accuracy)
        self.testing_acc_detail.append({
            'prob': prob,
            'action': action,
            'true_label': info['true_label'],
        })
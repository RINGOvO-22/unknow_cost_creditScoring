from collections import deque
import gymnasium as gym
import numpy as np
from scipy.special import expit

"""
Version 5: with 2D data generated
Similarly, use the data processed by methods from "Performative Prediction"
"""

# default hyperparameters
default_learning_rate = 1e-3
default_discount_factor = 0.99
default_predict_label_threshold = 0.5
default_batch_size = 128 # 32
default_init_cost_pram = 2.0

clipVal_td = 10.0 # for clipping the TD error
clipVal_grad_log_pi = 10.0 # for clipping the gradient of log policy
clipVal_policyWeight = 10.0 # for clipping the policy weight update

dimension = 2 # number of features
np.random.seed(0)

# a principal for the credit scoring v1 environment
class Principal_v5:
    def __init__(
        self,
        env: gym.Env,
        dimension: int = dimension,
        learning_rate_critic: float = default_learning_rate,
        learning_rate_actor: float = default_learning_rate,
        learning_rate_cost: float = default_learning_rate,
        buffer_size: int = default_batch_size,
        discount_factor: float = 0.99,
        init_cost_pram:  float = 2.0, # same as in the "made practical" paper 
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.lr_a = learning_rate_actor

        # hyperparameter: initial policy weight for the classifier
        self.previous_policy_weight = np.random.normal(loc=0.0, scale=0.1, size=(dimension+1,))
        # self.previous_policy_weight = np.ones(10+1, dtype=np.float64) * 0.01

        self.lr_c = learning_rate_critic
        # q value weights for the classifier (v*(s, a) + b)
        # +1: bias term,  +1: action term
        self.q_weights = np.ones(dimension+1+1, dtype=np.float64) * 0.01

        # initial cost parameter estimation for the principal (not used yet)
        self.cost_pram_estimation = np.full(shape=dimension, fill_value=init_cost_pram, dtype=np.float64)
        self.lr_cost = learning_rate_cost

        # buff
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

        # record training and testing process
        self.batch_update_count = 0
        self.training_error = []
        self.training_expected_acc_list = []
        self.training_acc_detail = []
        self.training_rewards = []
        self.training_policy_weights = []
        self.training_single_policy_weight_update = []
        self.training_batch_acc = []

        self.testing_accuracy = []
        self.testing_acc_detail = []

    # policy function
    def get_action(self, obs: np.ndarray, stochastic=True) -> int:
        logits = np.dot(self.previous_policy_weight, obs)
        # prob = 1 / (1 + np.exp(-logits))
        prob = expit(logits) # 相当于稳定的 1/(1+exp(-logits))
        
        if stochastic:
            action = np.random.binomial(n=1, p=prob)
        else:
            action = 1 if prob > default_predict_label_threshold else 0

        return prob, action

    def batch_update(self):
        if len(self.buffer) < self.buffer_size:
            return

        self.batch_update_count += 1
        batch = list(self.buffer)
        np.random.shuffle(batch)
        accs = []  # 本次 batch 的准确率记录

        for obs, action, reward, terminated, next_obs, true_label in batch:
            # obs_with_bias = np.append(obs, 1.0)
            q_input = np.append(obs, action)
            q_value = np.dot(self.q_weights, q_input)

            if not terminated and next_obs is not None:
                # next_obs_with_bias = np.append(next_obs, 1.0)
                q_next_0 = np.dot(self.q_weights, np.append(next_obs, 0))
                q_next_1 = np.dot(self.q_weights, np.append(next_obs, 1))
                max_q_next = max(q_next_0, q_next_1)
            else:
                max_q_next = 0.0

            td_target = reward + self.discount_factor * max_q_next
            td_error = td_target - q_value
            td_error = np.clip(td_error, -clipVal_td, +clipVal_td)

            self.q_weights += self.lr_c * td_error * q_input

            logits = np.dot(self.previous_policy_weight, obs)
            prob = 1 / (1 + np.exp(-logits))
            grad_log_pi = (action - prob) * obs
            # clip
            grad_log_pi = np.clip(grad_log_pi, -clipVal_grad_log_pi, clipVal_grad_log_pi)
            weight_update = (self.lr_a * td_error * grad_log_pi) / self.buffer_size
            weight_update = np.clip(weight_update, -clipVal_policyWeight, +clipVal_policyWeight)

            self.previous_policy_weight += weight_update

            self.training_single_policy_weight_update.append(weight_update)

            # 每步正确的概率(expected accuracy)
            if true_label == 1:
                correct_prob = prob
            else:
                correct_prob = 1 - prob
            accs.append(correct_prob)

        self.buffer.clear()
        self.training_batch_acc.append(np.mean(accs))  # 记录当前 batch 平均 accuracy

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

        self.buffer.append(sample)  # 单一 buffer

        if len(self.buffer) >= self.buffer_size:
            self.batch_update()

        # 记录单步结果
        # 用确定性策略评估预测
        pred = action
        # 每步正确的概率
        if true_label == 1:
            correct_prob = prob
        else:
            correct_prob = 1 - prob
        self.training_expected_acc_list.append(correct_prob)

        self.training_error.append(abs(prob - info['true_label']))
        self.training_rewards.append(reward)
        self.training_policy_weights.append(self.previous_policy_weight.copy())
        self.training_acc_detail.append({
            'predicted_prob': prob,
            'predicted_label': pred,
            'true_label': info['true_label'],
            'expected_accuracy': correct_prob,
            'reward': reward
        })

    def test_result_record(self, action: int, info: dict, prob: float):
        """
        记录测试结果
        参数:
            action: 执行的动作
            info: 包含 true_label 的字典
        """
        accuracy = 1.0 if action == int(info['true_label']) else 0.0
        self.testing_accuracy.append(accuracy)
        # detail record
        self.testing_acc_detail.append({
            'prob': prob,
            'action': action, 
            'true_label': info['true_label'],
            })
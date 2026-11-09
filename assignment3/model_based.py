"""从交互经验估计转移和奖励，再在学到的模型上做价值迭代。"""


class ModelBasedAgent:
    """Model-Based RL Protocol v1 —— 课程评测程序将按此接口加载并调用你的智能体。"""

    def __init__(self, states, actions, gamma: float = 0.9, seed=None):
        """states 为状态名列表，actions 为动作列表。"""
        self.states = list(states)
        self.actions = list(actions)
        self.gamma = float(gamma)
        self.seed = seed
        self._V = {state: 0.0 for state in self.states}
        self.iterations_ = 0
        self.delta_ = None
        self._set_unknown_model()

    def _set_unknown_model(self):
        """没有样本的 (s, a) 使用均匀转移、奖励 0。"""
        n_states = len(self.states)
        uniform = 1.0 / n_states if n_states else 0.0
        self._P = {
            state: {
                action: {next_state: uniform for next_state in self.states}
                for action in self.actions
            }
            for state in self.states
        }
        self._R = {
            state: {action: 0.0 for action in self.actions}
            for state in self.states
        }

    def learn_model(self, episodes: list):
        """
        从经验中估计模型。

        :param episodes: 列表的列表；每个 episode 是若干
                         (state, action, reward, next_state, done) 五元组。
        估计：
          P̂(s'|s,a) = N(s,a,s') / N(s,a)   （状态转移计数归一化）
          R̂(s,a)   = 该 (s,a) 上所有 reward 的样本均值
        对从未访问过的 (s,a)：P̂ 均匀分布，R̂ = 0。
        """
        counts = {
            state: {
                action: {next_state: 0 for next_state in self.states}
                for action in self.actions
            }
            for state in self.states
        }
        reward_sum = {
            state: {action: 0.0 for action in self.actions}
            for state in self.states
        }
        visits = {
            state: {action: 0 for action in self.actions}
            for state in self.states
        }

        for episode in episodes:
            for state, action, reward, next_state, _done in episode:
                counts[state][action][next_state] += 1
                reward_sum[state][action] += float(reward)
                visits[state][action] += 1

        n_states = len(self.states)
        for state in self.states:
            for action in self.actions:
                n = visits[state][action]
                if n == 0:
                    uniform = 1.0 / n_states
                    self._P[state][action] = {
                        next_state: uniform for next_state in self.states
                    }
                    self._R[state][action] = 0.0
                else:
                    self._P[state][action] = {
                        next_state: counts[state][action][next_state] / n
                        for next_state in self.states
                    }
                    self._R[state][action] = reward_sum[state][action] / n

    def _q_value(self, state, action, values):
        expected_next = 0.0
        for next_state, prob in self._P[state][action].items():
            expected_next += prob * values[next_state]
        return self._R[state][action] + self.gamma * expected_next

    def run_value_iteration(self, theta: float = 1e-4, max_iterations: int = 1000):
        """在学到的 P̂/R̂ 上执行同步扫描的价值迭代，直到收敛或达到 max_iterations。"""
        values = {state: 0.0 for state in self.states}
        delta = 0.0
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            updated = {}
            delta = 0.0
            for state in self.states:
                best = self._q_value(state, self.actions[0], values)
                for action in self.actions[1:]:
                    q = self._q_value(state, action, values)
                    if q > best:
                        best = q
                updated[state] = best
                change = abs(best - values[state])
                if change > delta:
                    delta = change
            values = updated
            if delta < theta:
                break
        self._V = values
        self.iterations_ = iteration
        self.delta_ = delta

    def get_transition(self, state, action) -> dict:
        """返回你估计的 {next_state: prob}。"""
        return dict(self._P[state][action])

    def get_reward(self, state, action) -> float:
        """返回你估计的 R̂(s,a)。"""
        return float(self._R[state][action])

    def get_value(self, state) -> float:
        return float(self._V[state])

    def get_policy(self, state):
        """返回 argmax_a Q̂(s,a)；平局时取动作列表中的第一个动作。"""
        best_action = self.actions[0]
        best_q = self._q_value(state, best_action, self._V)
        for action in self.actions[1:]:
            q = self._q_value(state, action, self._V)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

"""表格 Q-Learning。离线 learn() 按给定顺序更新，不在其中采样动作。"""


class QLearningAgent:
    """Q-Learning Protocol v1 —— 课程评测程序将按此接口加载并调用你的智能体。"""

    def __init__(self, states, actions, gamma: float = 0.9, alpha: float = 0.5,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05, seed=None):
        """epsilon 相关参数仅用于报告的附加在线探索实验；离线主任务不使用。"""
        self.states = list(states)
        self.actions = list(actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.seed = seed
        self._Q = {
            state: {action: 0.0 for action in self.actions}
            for state in self.states
        }

    def learn(self, episodes: list, mode: str = "fixed"):
        """
        从统一数据预算的 episodes 中按顺序做表格 Q-Learning 更新：
        逐条 episode、逐条转移调用 self.update()，不做任何动作采样。
        mode 参数保留接口一致性，本作业离线主任务中不使用探索率。
        """
        del mode
        for episode in episodes:
            for state, action, reward, next_state, done in episode:
                self.update(state, action, reward, next_state, bool(done))

    def update(self, state, action, reward, next_state, done: bool):
        """单步表格 Q-Learning 更新；done=True 时 target 中不含未来项。"""
        reward = float(reward)
        if done:
            target = reward
        else:
            next_values = self._Q[next_state]
            max_next = next_values[self.actions[0]]
            for next_action in self.actions[1:]:
                q = next_values[next_action]
                if q > max_next:
                    max_next = q
            target = reward + self.gamma * max_next
        previous = self._Q[state][action]
        self._Q[state][action] = (1.0 - self.alpha) * previous + self.alpha * target

    def get_q_value(self, state, action) -> float:
        return float(self._Q[state][action])

    def get_policy(self, state):
        """返回当前贪心动作（Q 表每行 argmax）。平局时取动作列表中的第一个。"""
        best_action = self.actions[0]
        best_q = self._Q[state][best_action]
        for action in self.actions[1:]:
            q = self._Q[state][action]
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

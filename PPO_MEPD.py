import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR

class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 卷积层（替代前两个全连接层）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=int(hidden_dim/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(hidden_dim/2), out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 全连接层（保持原结构）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, return_logits=False):
        # 检查输入维度并重塑为图像格式
        if x.dim() == 2:  # [num_agents, 3]
            grid_size = int(np.sqrt(x.size(0)))
            x = x.view(grid_size, grid_size, -1).permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 3:
            grid_size = int(np.sqrt(x.size(1)))
            x = x.view(x.shape[0], grid_size, grid_size, -1).permute(0, 3, 1, 2)
        elif x.dim() == 4:  # [batch, H, W, 3]
            x = x.permute(0, 3, 1, 2)
        
        # 卷积层处理
        x = self.conv_layers(x)  # [1,3,200,200]=[] [batch, hidden_dim, L, L]
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        x = x.permute(0, 2, 1)
        # 全连接层
        fc_out = self.fc(x)  # [batch, hidden_dim]
        
        # 输出层
        logits = self.actor(fc_out)  # [batch, 2]
        state_value = self.critic(fc_out).squeeze()  # [batch]
        
        if return_logits:
            return logits, state_value
        
        action_probs = F.softmax(logits, dim=-1)
        return action_probs, state_value

class SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, 
                    now_time, question, ppo_epochs, batch_size, gae_lambda,
                    output_path, delta, rho, 
                    expert_paths=None, tau=1.0, eta=0.1, beta=100):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time
        
        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta=delta # w_cl
        self.rho=rho # w_ent

        self.output_path=output_path
        
        # 神经网络
        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)  # 每1000步学习率降低10%
        def lower_bound_lr(epoch, min_lr=1e-4, step_size=100, gamma=0.5):
            factor = gamma ** (epoch // step_size)
            return max(factor, min_lr / initial_lr)  # 假设initial_lr是初始学习率

        # 初始化时需要知道初始学习率
        initial_lr = alpha  # 你的初始学习率
        self.scheduler = LambdaLR(self.optimizer, 
                                lr_lambda=lambda epoch: lower_bound_lr(epoch, min_lr=1e-3))
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # 策略蒸馏参数
        self.tau = tau
        self.eta = eta
        self.experts = []
        self.beta=beta
        
        # 加载专家模型
        if expert_paths is not None:
            for path in expert_paths:
                expert_model = self._load_expert(path)
                self.experts.append(expert_model)
        
    def _load_expert(self, path):
        """加载预训练的专家模型"""
        expert = ActorCritic().to(self.device)
        checkpoint = torch.load(path)
        expert.load_state_dict(checkpoint['model_state_dict'])
        expert.eval()  # 设置为评估模式
        return expert

    def _init_state(self, question):
        if question == 1: # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2: # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵
            state = torch.zeros((self.L_num, self.L_num))
            # 填充交替的 0 和 1
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 使用正确的填充参数格式 (padding_left, padding_right, padding_top, padding_bottom)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')  # 四周各填充1
        
        # 计算邻域合作数
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L, L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def calculate_reward(self, state_matrix):
        """计算每个智能体参与的5组博弈的总收益"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 2. 计算每个位置的邻域合作者数量（4邻居）
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        # 3. 计算中心智能体作为合作者时的单组收益 (r*n_C/5 - 1)
        c_single_profit = (self.r * neighbor_coop / 5) - 1
        
        # 4. 计算中心智能体作为背叛者时的单组收益 (r*n_C/5)
        d_single_profit = (self.r * neighbor_coop / 5)
        
        # 5. 对单组收益矩阵进行padding处理
        padded_c_profit = F.pad(c_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        padded_d_profit = F.pad(d_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 6. 计算每个智能体参与的5组博弈总收益
        # 中心智能体参与的5组博弈：自身作为中心的1组 + 作为邻居参与的4组
        c_total_profit = F.conv2d(padded_c_profit, self.neibor_kernel).squeeze() + c_single_profit
        d_total_profit = F.conv2d(padded_d_profit, self.neibor_kernel).squeeze() + d_single_profit
        
        # 7. 根据当前策略选择对应的总收益
        reward_matrix = torch.where(state_matrix.bool(), c_total_profit, d_total_profit)
        
        return reward_matrix

    def ppo_update(self):
        # 准备数据
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)
        
        # 转换为正确形状 (batch_size, L_num*L_num, 3)
        states = states.view(-1, self.L_num*self.L_num, 3)
        next_states = next_states.view(-1, self.L_num*self.L_num, 3)
        actions = actions.view(states.shape[0], self.L_num, self.L_num)
        rewards = rewards.view(states.shape[0], self.L_num, self.L_num)
        dones = dones.view(states.shape[0], self.L_num, self.L_num)
        old_log_probs = old_log_probs.view(states.shape[0], self.L_num, self.L_num)

        # 计算当前状态和下一状态的价值
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
        
        # 重塑价值函数输出匹配奖励形状
        values = values.reshape(rewards.shape[0], self.L_num, self.L_num)
        next_values = next_values.reshape(rewards.shape[0], self.L_num, self.L_num)

        # 计算GAE优势函数
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        # PPO更新循环
        for _ in range(self.ppo_epochs):
            # 创建随机批次
            perm = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                idx = perm[i:i+self.batch_size]
                state_b = states[idx]
                action_b = actions[idx]
                old_log_b = old_log_probs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]
                
                # 获取学生模型的预测
                probs, value_pred = self.policy(state_b)
                
                # 重塑概率和动作
                flat_probs = probs.view(-1, 2)  # [batch_size * L_num * L_num, 2]
                flat_action_b = action_b.reshape(-1)  # [batch_size * L_num * L_num]
                
                # 创建分布并计算对数概率
                dist = Categorical(flat_probs)
                log_probs = dist.log_prob(flat_action_b).view_as(action_b)
                
                # 计算熵
                entropy = dist.entropy().view_as(action_b).mean()
                
                # 计算PPO损失
                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 重塑critic值预测匹配返回形状
                value_pred = value_pred.reshape(ret_b.shape[0], ret_b.shape[1], ret_b.shape[2])
                critic_loss = F.mse_loss(value_pred, ret_b)
                
                # 初始化蒸馏损失
                distill_loss = 0
                
                # 策略蒸馏：从专家模型中学习
                if hasattr(self, 'epoch') and self.epoch < self.beta and len(self.experts) > 0:
                    # 获取学生模型的logits（未经softmax的输出）
                    student_logits, _ = self.policy(state_b, return_logits=True)
                    
                    # 收集所有专家的logits
                    all_expert_logits = []
                    for expert in self.experts:
                        with torch.no_grad():
                            expert_logits, _ = expert(state_b, return_logits=True)
                            all_expert_logits.append(expert_logits)
                    
                    # 计算专家logits的平均值
                    avg_expert_logits = torch.mean(torch.stack(all_expert_logits), dim=0)
                    
                    # 应用温度缩放
                    student_probs = F.log_softmax(student_logits / self.tau, dim=-1)
                    expert_probs = F.softmax(avg_expert_logits / self.tau, dim=-1)
                    
                    # 计算KL散度
                    distill_loss = F.kl_div(
                        student_probs, 
                        expert_probs, 
                        reduction='batchmean',
                        log_target=False
                    ) * (self.tau ** 2)
                
                # 组合所有损失项
                loss = actor_loss + self.delta * critic_loss - self.rho * entropy + \
                    self.eta * distill_loss
                
                self.rho=self.rho*0.995
                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()  # 更新学习率
        
        # 清空经验缓冲区
        self._reset_buffer()

    def _make_batch(self, states, actions, old_log_probs, advantages, returns):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i+self.batch_size]
            yield (states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx])

    def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch=epoch
            action, log_prob = self.choose_action(self.current_state)
            # next_state, reward, done = self.env_step(action)
            next_state=action
            reward = self.calculate_reward(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)
            
            self.states.append(self.encode_state(self.current_state).view(self.L_num,self.L_num,3))
            self.actions.append(action)
            self.log_probs.append(log_prob)

            self.rewards.append(reward)
            self.next_states.append(self.encode_state(next_state).view(self.L_num,self.L_num,3))
            self.dones.append(done)
            
            if epoch==0:
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch, self.r, profit_matrix))
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)
            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()  # PPO 先更新
                self.current_state = next_state
                self._reset_buffer()  # 清空缓冲区
            else:
                # 如果没有达到更新条件，直接更新当前状态
                self.current_state = next_state           

            # self.current_state = next_state
            # 在关键时间点保存快照（与原Q-learning相同）
            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch+1, self.r, profit_matrix))
                # self.save_checkpoint()

            if epoch % 1000 == 0:
                self.save_checkpoint()
            # 收集数据

            # 收集数据
            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()
            
            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)
        
        self.save_checkpoint(is_final=True)

        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        # output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    # def mkdir(self, path):
    #     os.makedirs(path, exist_ok=True)
    
    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略矩阵快照与数据文件（与原Q-learning代码相同格式）"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        # img_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        # matrix_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        # profit_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # =============================================
        # 1. 保存策略矩阵图
        # =============================================
        # 生成策略矩阵可视化
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        ax1.axis('off')
        # 设置 figure 的边框为黑色
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)  # 设置边框线宽
        
        # 创建颜色映射（黑:背叛者，白:合作者）
        color_map = {
            0: [0, 0, 0],    # 黑色
            1: [1, 1, 1]     # 白色
        }
        
        # 转换为RGB图像
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color
        
        # 绘图设置
        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
            
        # 保存图片
        # plt.savefig(f'{img_dir}/t={epoch}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
        
        # =============================================
        # 2. 保存收益热图
        # =============================================
        # 2. 处理profit数据
        if isinstance(profit_data, tuple):
            # 如果是元组，提取combined_reward和team_utility
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data
        
        # 确保是张量并转移到CPU
        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        # 绘制热图
        vmin = 0
        vmax = np.ceil(np.maximum(5*(r-1),4*r))
        # profit_data = profit_matrix.cpu().numpy()
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')
        
        # 添加颜色条
        
        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=28)
        # cbar2.set_ticks(np.arange(vmin, vmax, 1))
        # cbar.set_label('Profit Value')
        cbar2.set_ticks(np.ceil(np.linspace(vmin, vmax, 5)).astype(int))
        
        # 设置坐标轴
        ax2.set_xticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.set_yticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.grid(False)

        ax2.axis('off')
        # 设置 figure 的边框为黑色
        # fig2.patch.set_edgecolor('black')
        # fig2.patch.set_linewidth(2)  # 设置边框线宽
        
        # 保存收益热图
        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

        # =============================================
        # 3. 保存矩阵数据文件
        # =============================================
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix, fmt='%.4f')
        return 0

    def _reset_buffer(self):
        """显式释放显存"""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        torch.cuda.empty_cache()  # 立即释放未使用的显存
    
    # 修改经验存储逻辑
    def _store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储时分离梯度"""
        self.states.append(state.detach().cpu())  # 转移到CPU
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())

    def save_checkpoint(self, is_final=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'r': self.r,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
        }
        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")
    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state(state_matrix)
            logits, _ = self.policy(features, return_logits=True)  # 获取logits
            probs = F.softmax(logits, dim=-1)  # 手动计算概率
            dist = Categorical(probs)
            actions = dist.sample()
        return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from agent.network import Scheduler
from torch_geometric.data import Batch


class Agent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_actor_layers,  # actor layer의 개수
                 num_critic_layers,  # critic layer의 개수
                 lr,  # 학습률
                 lr_decay,  # 학습률에 대한 감소비율
                 lr_step,  # 학습률 감소를 위한 스텝 수
                 gamma,  # 감가율
                 lmbda,  # gae 파라미터
                 eps_clip,  # loss function 내 clipping ratio
                 K_epoch,  # 동일 샘플에 대한 update 횟수
                 P_coeff,  # 정책 학습에 대한 가중치
                 V_coeff,  # 가치함수 학습에 대한 가중치
                 E_coeff,  # 엔트로피에 대한 가중치
                 use_gnn=True,
                 use_added_info=True,
                 device="cpu"):

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.P_coeff = P_coeff
        self.V_coeff = V_coeff
        self.E_coeff = E_coeff
        self.device = device

        self.data = []
        self.network = Scheduler(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                 num_HGT_layers, num_actor_layers, num_critic_layers,
                                 use_gnn=use_gnn, use_added_info=use_added_info).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=lr_step, gamma=lr_decay)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, info_lst, a_lst, r_lst, s_prime_lst, info_prime_lst, a_logprob_lst, v_lst, mask_lst, current_ops_lst, done_lst \
            = [], [], [], [], [], [], [], [], [], [], []

        for i, transition in enumerate(self.data):
            s, info, a, r, s_prime, info_prime, a_logprob, v, mask, current_ops, done = transition

            s_lst.append(s)
            info_lst.append(info.unsqueeze(0))
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            info_prime_lst.append(info_prime.unsqueeze(0))
            a_logprob_lst.append([a_logprob])
            if i > 0:
                v_lst.append([v])
            mask_lst.append(mask.unsqueeze(0))
            current_ops_lst.append(current_ops.unsqueeze(0))
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        if done_lst[-1] == 0:
            v_lst.append([0.0])
        else:
            with torch.no_grad():
                _, _, v = self.network.act(s_prime_lst[-1], mask, current_ops, info_prime)
            v_lst.append([v])

        s, info, a, r, s_prime, info_prime, a_logprob, v, mask, current_ops, done \
            = (Batch.from_data_list(s_lst).to(self.device),
               torch.concat(info_lst).to(self.device),
               torch.tensor(a_lst).to(self.device),
               torch.tensor(r_lst, dtype=torch.float).to(self.device),
               Batch.from_data_list(s_prime_lst).to(self.device),
               torch.concat(info_prime_lst).to(self.device),
               torch.tensor(a_logprob_lst).to(self.device),
               torch.tensor(v_lst, dtype=torch.float).to(self.device),
               torch.concat(mask_lst).to(self.device),
               torch.concat(current_ops_lst).to(self.device),
               torch.tensor(done_lst, dtype=torch.float).to(self.device))

        self.data = []

        return s, info, a, r, s_prime, info_prime, a_logprob, v, mask, current_ops, done

    def get_action(self, s, mask, current_ops, added_info):
        self.network.eval()
        with torch.no_grad():
            a, log_p, v = self.network.act(s, mask, current_ops, added_info)
        return a, log_p, v

    def train(self):
        self.network.train()
        s, info, a, r, s_prime, info_prime, a_logprob, v, mask, current_ops, done = self.make_batch()
        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = r + self.gamma * v * done
            delta = td_target - v

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta.flip(dims=(0,)):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.concat(advantage_lst).unsqueeze(-1).to(self.device)

            new_a_logprob, new_v, dist_entropy = self.network.evaluate(s, a, mask, current_ops, info)
            ratio = torch.exp(new_a_logprob - a_logprob)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = - self.P_coeff * torch.min(surr1, surr2) + self.V_coeff * F.smooth_l1_loss(new_v, td_target) - self.E_coeff * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)
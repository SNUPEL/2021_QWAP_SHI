import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch.distributions import Categorical
from torch.nn import Parameter


class Scheduler(nn.Module):
    def __init__(self, meta_data, state_size, num_nodes, embed_dim, num_heads,
                       num_HGT_layers, num_actor_layers, num_critic_layers, use_gnn=True, use_added_info=True):
        super(Scheduler, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_actor_layers = num_actor_layers
        self.num_critic_layers = num_critic_layers
        self.use_gnn = use_gnn
        self.use_added_info = use_added_info

        if use_gnn:
            self.conv = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
                else:
                    self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))
        else:
            self.mlp_quay = nn.ModuleList()
            self.mlp_operation = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.mlp_quay.append(nn.Linear(self.state_size["quay"], embed_dim))
                    self.mlp_operation.append(nn.Linear(self.state_size["operation"], embed_dim))
                else:
                    self.mlp_quay.append(nn.Linear(embed_dim, embed_dim))
                    self.mlp_operation.append(nn.Linear(embed_dim, embed_dim))

        if use_added_info:
            self.fc = nn.ModuleList()
            self.fc.append(nn.Linear(2, embed_dim))
            self.fc.append(nn.Linear(embed_dim, embed_dim))

        if use_added_info:
            concat_dim = embed_dim * 3
        else:
            concat_dim = embed_dim * 2

        self.actor = nn.ModuleList()
        for i in range(num_actor_layers):
            if i == 0:
                self.actor.append(nn.Linear(concat_dim, embed_dim))
            elif 0 < i < num_actor_layers - 1:
                self.actor.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.actor.append(nn.Linear(embed_dim, 1))

        self.critic = nn.ModuleList()
        for i in range(num_critic_layers):
            if i == 0:
                self.critic.append(nn.Linear(embed_dim * 2, embed_dim))
            elif i < num_critic_layers - 1:
                self.critic.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.critic.append(nn.Linear(embed_dim, 1))

    def act(self, state, mask, current_ops, added_info, greedy=False):
        x_dict, edge_index_dict = state.x_dict, state.edge_index_dict

        if self.use_gnn:
            for i in range(self.num_HGT_layers):
                x_dict = self.conv[i](x_dict, edge_index_dict)
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}

            h_quays = x_dict["quay"]
            if "operation" in self.meta_data[0]:
                h_ops = x_dict["operation"]
            elif "ship" in self.meta_data[0]:
                h_ships = x_dict["ship"]
        else:
            h_quays = x_dict["quay"]
            h_ops = x_dict["operation"]
            for i in range(self.num_HGT_layers):
                h_quays = self.mlp_quay[i](h_quays)
                h_quays = F.elu(h_quays)
                h_ops = self.mlp_operation[i](h_ops)
                h_ops = F.elu(h_ops)

        h_quays_pooled = h_quays.mean(dim=-2)
        if "operation" in self.meta_data[0]:
            h_ops_pooled = h_ops.mean(dim=-2)
            ships_gather = current_ops.unsqueeze(-1).expand(-1, self.embed_dim)
            h_ships = h_ops.gather(0, ships_gather)
        elif "ship" in self.meta_data[0]:
            h_ships_pooled = h_ships.mean(dim=-2)

        h_ships_padding = h_ships.unsqueeze(-2).expand(-1, self.num_nodes["quay"], -1)
        h_quays_padding = h_quays.unsqueeze(-3).expand_as(h_ships_padding)

        # h_quays_pooled_padding = h_quays_pooled[None, None, :].expand_as(h_quays_padding)
        # h_ships_pooled_padding = h_ships_pooled[None, None, :].expand_as(h_ships_padding)

        if self.use_added_info:
            h_added = added_info
            for i in range(self.num_HGT_layers):
                h_added = self.fc[i](h_added)
                h_added = F.elu(h_added)
            h_actions = torch.cat((h_quays_padding, h_ships_padding, h_added), dim=-1)
        else:
            h_actions = torch.cat((h_quays_padding, h_ships_padding), dim=-1)

        if "operation" in self.meta_data[0]:
            h_pooled = torch.cat((h_quays_pooled, h_ops_pooled), dim=-1)
        elif "ship" in self.meta_data[0]:
            h_pooled = torch.cat((h_quays_pooled, h_ships_pooled), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                logits = self.actor[i](h_actions).flatten()

        mask = mask.transpose(0, 1).flatten()
        logits[~mask] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if greedy:
            action = torch.argmax(probs)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                state_value = self.critic[i](h_pooled)

        return action.item(), action_logprob.item(), state_value.squeeze().item()

    def evaluate(self, batch_state, batch_action, batch_mask, batch_current_ops, batch_added_info):
        batch_size = batch_state.num_graphs
        x_dict, edge_index_dict = batch_state.x_dict, batch_state.edge_index_dict

        if self.use_gnn:
            for i in range(self.num_HGT_layers):
                x_dict = self.conv[i](x_dict, edge_index_dict)
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}

            h_quays = x_dict["quay"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            if "operation" in self.meta_data[0]:
                h_ops = x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            elif "ship" in self.meta_data[0]:
                h_ships = x_dict["ship"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        else:
            h_quays = batch_state["quay"]['x']
            h_ops = batch_state["operation"]['x']
            for i in range(self.num_HGT_layers):
                h_quays = self.mlp_quay[i](h_quays)
                h_quays = F.elu(h_quays)
                h_ops = self.mlp_operation[i](h_ops)
                h_ops = F.elu(h_ops)

            h_quays = h_quays.unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_ops = h_ops.unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        h_quays_pooled = h_quays.mean(dim=-2)
        if "operation" in self.meta_data[0]:
            h_ops_pooled = h_ops.mean(dim=-2)
            ships_gather = batch_current_ops.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            h_ships = h_ops.gather(1, ships_gather)
        elif "ship" in self.meta_data[0]:
            h_ships_pooled = h_ships.mean(dim=-2)

        h_ships_padding = h_ships.unsqueeze(-2).expand(-1, -1, self.num_nodes["quay"], -1)
        h_quays_padding = h_quays.unsqueeze(-3).expand_as(h_ships_padding)

        # h_quays_pooled_padding = h_quays_pooled[:, None, None, :].expand_as(h_quays_padding)
        # h_ships_pooled_padding = h_ships_pooled[:, None, None, :].expand_as(h_ships_padding)

        if self.use_added_info:
            h_added = batch_added_info
            for i in range(self.num_HGT_layers):
                h_added = self.fc[i](h_added)
                h_added = F.elu(h_added)
            h_actions = torch.cat((h_quays_padding, h_ships_padding, h_added), dim=-1)
        else:
            h_actions = torch.cat((h_quays_padding, h_ships_padding), dim=-1)

        if "operation" in self.meta_data[0]:
            h_pooled = torch.cat((h_quays_pooled, h_ops_pooled), dim=-1)
        elif "ship" in self.meta_data[0]:
            h_pooled = torch.cat((h_quays_pooled, h_ships_pooled), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                batch_logits = self.actor[i](h_actions).flatten(1)

        batch_mask = batch_mask.transpose(1, 2).flatten(1)
        batch_logits[~batch_mask] = float('-inf')
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_dist = Categorical(batch_probs)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                batch_state_values = self.critic[i](h_pooled)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys
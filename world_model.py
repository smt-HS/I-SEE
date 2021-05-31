from agenda import UserAgenda
import torch
from utils import usr_action_vectorize, usr_state_vectorize, to_device
from copy import deepcopy
from rlmodule import MultiDiscretePolicy, DiscretePolicy, Value
import logging
import os
from collections import namedtuple
from pprint import pprint
from agenda import Goal, GoalGenerator, Agenda
from utils import init_session, init_goal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WorldModel(UserAgenda):

    def __init__(self, args, config, manager, load_dataset=False):
        super(WorldModel, self).__init__(args.data_dir, config)
        self.ensemble_size = args.ensemble_size
        self.index = None
        config_tuple = namedtuple('config', ('s_dim', 'h_dim', 'a_dim'))
        policy_config = config_tuple(config.s_dim_usr, config.h_dim, config.a_dim_usr)

        self.policies = [MultiDiscretePolicy(policy_config).to(DEVICE) for _ in range(self.ensemble_size)]

        # self.value = Value(config)
        terminal_config = config_tuple(config.s_dim_usr, config.h_dim, 1)
        self.terminals = [MultiDiscretePolicy(terminal_config).to(DEVICE) for _ in range(self.ensemble_size)]

        if load_dataset:
            self.data_train = manager.create_dataset_usr('train', args.batchsz, config)
            self.data_valid = manager.create_dataset_usr('valid', args.batchsz, config)
            self.data_test = manager.create_dataset_usr('test', args.batchsz, config)
        else:
            self.data_train, self.data_valid, self.data_test = None, None, None

        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.tau = args.tau
        self.policy_optims = [torch.optim.RMSprop(policy.parameters(), lr=args.lr_rl) for policy in self.policies]
        # self.value_optim = torch.optim.Adam(self.value.parameters(), lr=args.lr_rl)
        self.terminal_optims = [torch.optim.Adam(terminal.parameters(), lr=args.lr_rl) for terminal in self.terminals]
        self.multi_entropy_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.binary_loss = torch.nn.BCEWithLogitsLoss()

        self.print_per_batch = args.print_per_batch
        self.save_per_epoch = args.save_per_epoch
        self.save_dir = args.save_dir
        self.optim_batchsz = args.batchsz
        self.update_round = args.update_round

        for policy in self.policies:
            policy.eval()
        # self.value.eval()

        for terminal in self.terminals:
            terminal.eval()

    def reset(self, random_seed=None):
        self.goal = Goal(self.goal_generator, self._mask_user_goal, seed=random_seed)
        self.agenda = Agenda(self.goal)

        dummy_state, dummy_goal = init_session(-1, self.cfg)
        init_goal(dummy_goal, self.goal.domain_goals, self.cfg)
        dummy_state['user_goal'] = dummy_goal
        dummy_state['last_user_action'] = dict()

        usr_a, terminal = self.predict(dummy_state, {})

        init_state = self.update_belief_usr(dummy_state, usr_a, terminal)
        return init_state

    def set_state(self, state, usr_act_vec, terminal):
        new_state = self.update_belief_usr(state, usr_act_vec, terminal)
        return new_state

    def set_goal_agenda(self, goal, agenda):
        self.goal = deepcopy(goal)
        self.agenda = deepcopy(agenda)

    def pick_one(self, index):
        self.index = index

    def step(self, state, sys_a):
        # update state with sys_act
        current_s = self.update_belief_sys(state, sys_a)

        # update the goal
        da_dict = self._action_to_dict(current_s['sys_action'])
        sys_action = self._transform_sysact_in(da_dict)
        self.agenda.update(sys_action, self.goal)

        if current_s['others']['terminal']:
            # user has terminated the session at last turn
            usr_a, terminal = torch.zeros(self.cfg.a_dim_usr, dtype=torch.int32), True
        else:
            # da_dict = self._action_to_dict(current_s['sys_action'])
            # pprint(current_s['sys_action'])
            usr_a, terminal = self.predict(state, current_s['sys_action'])

        # update state with user_act
        next_s = self.update_belief_usr(current_s, usr_a, terminal)
        return next_s, terminal, {}

    def predict(self, state, sys_action):
        # pprint(state)
        # pprint(sys_action)
        sys_action_list = list(state['history']['sys'].keys()) + list(sys_action.keys())

        # state_vec = usr_state_vectorize(self.goal.domain_goals, sys_action_list, self.cfg, from_log=False)

        state_vec = usr_state_vectorize(state['user_goal'], sys_action_list, self.cfg)

        state_vec = torch.Tensor(state_vec).to(DEVICE)
        user_action_vec = self.policies[self.index].select_action(state_vec, sample=True)
        terminal = self.terminals[self.index].select_action(state_vec, sample=True)
        terminal = bool(terminal)
        return user_action_vec, terminal

    def imitate_loop(self, data):
        s, target_a, target_terminal = to_device(data)
        a_weights = self.policies[self.index](s)
        loss_a = self.multi_entropy_loss(a_weights, target_a)

        t_weights = self.terminals[self.index](s)
        loss_t = self.binary_loss(t_weights, target_terminal)

        return loss_a, loss_t

    def imitating(self, epoch, dataloader=None):
        data_loader = dataloader or self.data_train
        self.policies[self.index].train()
        self.terminals[self.index].train()
        pi_loss, done_loss = 0., 0.
        for i, data in enumerate(data_loader):
            self.policy_optims[self.index].zero_grad()
            self.terminal_optims[self.index].zero_grad()

            loss_pi, loss_d = self.imitate_loop(data)

            pi_loss += loss_pi.item()
            loss_pi.backward()
            self.policy_optims[self.index].step()

            done_loss += loss_d.item()
            loss_d.backward()
            self.terminal_optims[self.index].step()

            if (i + 1) % self.print_per_batch == 0:
                pi_loss /= self.print_per_batch
                done_loss /= self.print_per_batch
                logging.debug('<<world model {}>> epoch {}, iter {}, loss_pi:{}, loss_done:{}'.format(
                    self.index, epoch, i, pi_loss, done_loss))
                pi_loss = 0.
                done_loss = 0.

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policies[self.index].eval()
        self.terminals[self.index].eval()

    def imit_test(self, epoch, best_pi, best_done):
        pi_loss, done_loss = 0., 0.
        for i, data in enumerate(self.data_valid):
            loss_pi, loss_d = self.imitate_loop(data)

            pi_loss += loss_pi.item()
            done_loss += loss_d.item()

        pi_loss /= len(self.data_valid)
        done_loss /= len(self.data_valid)

        logging.debug(
            '<<world model {}>> validation, epoch {}, loss_pi:{}, loss_done:{}'.format(self.index, epoch, pi_loss,
                                                                                       done_loss))
        if pi_loss < best_pi:
            logging.info('<<world model {}>> best policy model saved'.format(self.index))
            best_pi = pi_loss
            self.save(self.save_dir, 'best', save_policy=True, save_terminal=False)
        if done_loss < best_done:
            logging.info('<<world model {}>> best terminal model saved'.format(self.index))
            best_done = done_loss
            self.save(self.save_dir, 'best', save_policy=False, save_terminal=True)

        pi_loss, done_loss = 0., 0.
        for i, data in enumerate(self.data_test):
            loss_pi, loss_d = self.imitate_loop(data)

            pi_loss += loss_pi.item()
            done_loss += loss_d.item()

        pi_loss /= len(self.data_valid)
        done_loss /= len(self.data_valid)

        logging.debug(
            '<<world model {}>> test, epoch {}, loss_pi:{}, loss_done:{}'.format(self.index, epoch, pi_loss, done_loss))
        return best_pi, best_done

    def train(self, data, epoch):
        for i in range(self.ensemble_size):
            self.pick_one(i)

            self.policies[self.index].train()
            self.terminals[self.index].train()
            self.policy_optims[self.index].zero_grad()
            self.terminal_optims[self.index].zero_grad()

            loss_pi, loss_done = self.imitate_loop(data)

            logging.debug('<<world model {}>>, epoch {}, loss_pi:{}, loss_done:{}'.format(
                self.index, epoch, loss_pi.item(), loss_done.item()))

            loss_pi.backward()
            self.policy_optims[self.index].step()
            loss_done.backward()
            self.terminal_optims[self.index].step()

            self.policies[self.index].eval()
            self.terminals[self.index].eval()

    def test(self, data, epoch, best):
        for i in range(self.ensemble_size):
            self.pick_one(i)

            loss_pi, loss_done = self.imitate_loop(data)

            loss_pi, loss_done = loss_pi.item(), loss_done.item()

            if loss_pi < best['pi'][i]:
                logging.info('<<world model {}>>,  epoch {}, best policy model saved'.format(self.index, epoch))
                best['pi'][i] = loss_pi
                self.save(self.save_dir, 'best', save_policy=True, save_terminal=False)
            if loss_done < best['done'][i]:
                logging.info('<<world model {}>>,  epoch {}, best terminal model saved'.format(self.index, epoch))
                best['done'][i] = loss_done
                self.save(self.save_dir, 'best', save_policy=False, save_terminal=True)

            logging.debug(
                '<<world model {}>> test, epoch {}, loss_pi:{}, loss_done:{}'.format(self.index, epoch, loss_pi,
                                                                                     loss_done))
        return best

    def save(self, directory, epoch, save_policy=True, save_terminal=True):
        if not os.path.exists(directory):
            os.makedirs(directory)

        assert save_policy or save_terminal

        if save_policy:
            torch.save(self.policies[self.index].state_dict(),
                       directory + '/' + str(epoch) + '_wm_{}.pol.mdl'.format(self.index))
        if save_terminal:
            torch.save(self.terminals[self.index].state_dict(),
                       directory + '/' + str(epoch) + '_wm_{}.ter.mdl'.format(self.index))

        logging.info('<<world model {}>> epoch {}: saved network to mdl'.format(self.index, epoch))

    def load(self, filename):
        policy_mdl = filename + '_wm_{}.pol.mdl'.format(self.index)
        terminal_mdl = filename + '_wm_{}.ter.mdl'.format(self.index)

        if os.path.exists(policy_mdl):
            self.policies[self.index].load_state_dict(torch.load(policy_mdl))
            logging.info('<<world model {}>> loaded checkpoint from file: {}'.format(self.index, policy_mdl))
        if os.path.exists(terminal_mdl):
            self.terminals[self.index].load_state_dict(torch.load(terminal_mdl))
            logging.info('<<world model {}>> loaded checkpoint from file: {}'.format(self.index, terminal_mdl))

import numpy as np
import matplotlib.pyplot as plt


class ValueIteration:
    def __init__(self, reward_function, transition_model, CVaR, gamma):
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
        self.CVaR = CVaR
        self.values = np.zeros(self.num_states)
        self.policy = None

    def one_iteration(self):
        delta = 0
        #CVaR = [0.10154284,0.52817785,3.08546528,7.45201895,6.72745085,2.2829513,0.34587265,0.08870011,1.00435018,1.4309852,3.98827263,8.35482629,7.6302582,3.18575865,1.24868,0.99150746,4.63119822,5.05783324,7.61512067,15.98167433,11.25710624,6.81260669,4.87552804,4.6183555,7.99450879,8.4211438,10.97843123,15.3449849,14.6204168,10.17591726,8.2388386,7.98166606,5.09748933,5.52412435,8.08141178,12.44796544,11.72339735,7.2788978,5.34181914,5.0846466,1.21267931,1.63931432,4.19660175,8.56315542,7.83858732,3.39408777,1.45700912,1.19983658]
        for s in range(self.num_states):
            temp = self.values[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)*(max(self.CVaR)-self.CVaR[s])/max(self.CVaR) 
                

            self.values[s] = max(v_list)
            delta = max(delta, abs(temp - self.values[s]))
        return delta

    def get_policy(self):
        pi = np.ones(self.num_states) * -1
        #CVaR = [0.10154284,0.52817785,3.08546528,7.45201895,6.72745085,2.2829513,0.34587265,0.08870011,1.00435018,1.4309852,3.98827263,8.35482629,7.6302582,3.18575865,1.24868,0.99150746,4.63119822,5.05783324,7.61512067,15.98167433,11.25710624,6.81260669,4.87552804,4.6183555,7.99450879,8.4211438,10.97843123,15.3449849,14.6204168,10.17591726,8.2388386,7.98166606,5.09748933,5.52412435,8.08141178,12.44796544,11.72339735,7.2788978,5.34181914,5.0846466,1.21267931,1.63931432,4.19660175,8.56315542,7.83858732,3.39408777,1.45700912,1.19983658]
        for s in range(self.num_states):
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                p = self.transition_model[s, a]
                v_list[a] = self.reward_function[s] + self.gamma * np.sum(p * self.values)*(max(self.CVaR)-self.CVaR[s])/max(self.CVaR) 


            max_index = []
            max_val = np.max(v_list)
            for a in range(self.num_actions):
                if v_list[a] == max_val:
                    max_index.append(a)
            pi[s] = np.random.choice(max_index)
        return pi.astype(int)

    def train(self, tol=1e-3, plot=False):
        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]
        while delta > tol:
            epoch += 1
            delta = self.one_iteration()
            delta_history.append(delta)
            if delta < tol:
                break
        self.policy = self.get_policy()

        # print(f'# iterations of policy improvement: {len(delta_history)}')
        # print(f'delta = {delta_history}')

        if plot is True:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
            ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                    alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{self.gamma}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.legend()
            plt.tight_layout()
            plt.show()








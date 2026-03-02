import os
import numpy as np
import torch

class BayesianHMMSampler:
    def __init__(self, args, num_states=3, init=False, init_dir=None):
        """
        Initialize a Bayesian HMM with online learning of transition probabilities
        :param args: Configuration parameters
        :param total_num_samples: Total number of samples
        :param num_states: Number of states in the HMM (default: 3, corresponding to {1,2,3})
        :param prior_alpha: Concentration parameter for the Dirichlet prior
        :param init: Whether to initialize from external file
        :param init_dir: Path to the initialization file
        """
        self.args = args
        self.real_batch_size = self.args.data.train_batch_size
        self.num_states = num_states
        if isinstance(args.tasksampler.hmm_prior_alpha, float):
            self.prior_alpha = np.ones((self.num_states, self.num_states)) * args.tasksampler.hmm_prior_alpha
        elif args.tasksampler.hmm_prior_alpha == 'progress':
            self.prior_alpha = np.array([[1.0,0.5,0.5],
                                         [1.0,1.0,0.5],
                                         [1.0,1.0,1.0]])
        elif args.tasksampler.hmm_prior_alpha == 'stability':
            self.prior_alpha = np.array([[1.0,0.5,0.5],
                                         [0.5,1.0,0.5],
                                         [0.5,0.5,1.0]])
        elif args.tasksampler.hmm_prior_alpha == 'local':
            self.prior_alpha = np.array([[1.0,1.0,0.0],
                                         [1.0,1.0,1.0],
                                         [0.0,1.0,1.0]])
        else:
            raise ValueError("hmm_prior_alpha must be a float or specified string")

        if args.tasksampler.hmm_prior_pi == 'uniform':
            self.prior_pi = np.ones(self.num_states) / self.num_states
        elif args.tasksampler.hmm_prior_pi == '1':
            self.prior_pi = np.array([1.0, 0.0, 0.0])
        elif args.tasksampler.hmm_prior_pi == '2':
            self.prior_pi = np.array([0.0, 1.0, 0.0])
        elif args.tasksampler.hmm_prior_pi == '3':
            self.prior_pi = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError("hmm_prior_pi must be 'uniform', '1', '2', or '3'")

        # Maintain a separate state and transition matrix for each sample ID
        self.alphas = {}  # dictionary: index -> transition matrix Dirichlet parameters (shape [num_states, num_states])
        self.pis = {}     # dictionary: index -> current state distribution (shape [num_states])
        self.pis_before = {}  # dictionary: index -> previous state distribution (shape [num_states])
        self.sampled_times = {}     # dictionary: index -> number of times sampled (integer)
        self.learning_rate = 1.0 # default learning rate for Bayesian update
        self.sampling_strategy = args.tasksampler.hmm_sample_strategy #
        self.no_update = args.tasksampler.hmm_no_update
        self.hmm_transition_decay_ratio = args.tasksampler.hmm_transition_decay_ratio # decay ratio for transition parameters to handle non-stationarity
        # self.hmm_state_update_smooth_ratio = args.tasksampler.hmm_state_update_smooth_ratio
    
    def get_alpha_pi(self, index):
        """Get alpha and pi for the given index, initialize if not exist"""
        index = str(index)
        if index not in self.alphas:
            self.alphas[index] = self.prior_alpha
        if index not in self.pis:
            self.pis[index] = self.prior_pi
        if index not in self.pis_before:
            self.pis_before[index] = self.prior_pi
        if index not in self.sampled_times:
            self.sampled_times[index] = 0
        
        return self.alphas[index], self.pis[index], self.pis_before[index]
    
    def predict_next_state(self, alpha, pi):
        """
        Prediction - Calculate the prior distribution of the next state
        :param alpha: Dirichlet parameters of the transition matrix
        :param pi: Current state distribution
        :return: Prior probability distribution of the next state
        """
        # compute the expected transition matrix from the Dirichlet parameters
        P_expected = alpha / alpha.sum(axis=0, keepdims=True)
        # predict next state distribution
        pi_next_prior = np.dot(P_expected, pi)
        # ensure pi_next_prior is a valid probability distribution
        pi_next_prior /= pi_next_prior.sum()
        return pi_next_prior
    
    def update_transition_params(self, alpha, pi, pi_next, learning_rate=None):
        """
        Baysian update of transition parameters using soft counts
        :param alpha: Dirichlet parameters of the transition matrix
        :param pi: Current state distribution
        :param pi_next: State distribution at t+1
        :param learning_rate: Learning rate to control the update magnitude
        :return: Updated alpha
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        P_expected = alpha / alpha.sum(axis=0, keepdims=True)
        numerator = np.outer(pi_next, pi) * P_expected
        denominator = numerator.sum()  # normalization factor
        update = (numerator / denominator) * learning_rate
        updated_alpha = alpha * self.hmm_transition_decay_ratio + self.prior_alpha * (1 - self.hmm_transition_decay_ratio) + update
        return updated_alpha
    
    def sample_batch(self, batch_candidates_dict):
        """
        Sample a batch of samples from the candidate set based on the current state distribution and specified sampling strategy.
        :param batch_candidates_dict: A dictionary containing candidate samples, with keys such as 'index'
        :return: The sampled batch and their corresponding state distribution scores
        """
        candidate_indices = batch_candidates_dict['index']
        m = len(candidate_indices)
        assert self.real_batch_size <= m, "batch_size must be <= number of candidates"
        
        # compute state distribution and entropy for each candidate
        state_probs = []
        entropies = []
        
        for index in candidate_indices:
            alpha, pi, _ = self.get_alpha_pi(str(index))
            # if self.args.tasksampler.hmm_sample_untrained and self.sampled_times[str(index)] == 0 and self.sampling_strategy == 'topk':
            #     pi = np.array([0.0, 1.0, 0.0])
            state_probs.append(pi)
            # compute entropy as uncertainty measure
            entropy = -np.sum(pi * np.log(pi + 1e-10))
            entropies.append(entropy)
        state_probs = np.array(state_probs)  # [m, num_states]
        entropies = np.array(entropies)  # [m]

        # sample indices based on the specified sampling strategy
        if self.sampling_strategy == 'uniform':
            sampled_index = np.random.choice(m, size=self.real_batch_size, replace=False)
        elif self.sampling_strategy == 'topk':
            # select k samples with the highest second state probability
            sampled_index = np.argsort(-state_probs[:,1])[:self.real_batch_size]
        elif self.sampling_strategy == 'topk+entropy':
            # select k samples with the highest combined score of state probability and entropy
            combined_scores = state_probs[:,1] + entropies * self.args.tasksampler.hmm_entropy_weight
            sampled_index = np.argsort(-combined_scores)[:self.real_batch_size]
        
        batch_candidates_dict = {k: v[sampled_index] for k, v in batch_candidates_dict.items()}
        return batch_candidates_dict, torch.tensor(state_probs[sampled_index,:])
    
    def train(self, batch_candidates_dict, y):
        """
        Update HMM parameters based on the observed labels for the sampled batch.
        :param batch_candidates_dict: A dictionary containing the sampled batch, with keys such as 'index'
        :param y: Observed success rates for the sampled batch (values in [0,1], can be converted to state observations)
        :return: None, None, None (to maintain interface consistency with other samplers)
        """
        if self.no_update:
            return None, None, None
        indices = batch_candidates_dict['index']
        # For idx in indices, self.alpha and self.pis are updated as follows
        for idx, s in zip(indices, y):
            idx_str = str(idx)
            alpha, pi, pi_before = self.get_alpha_pi(idx_str)
            # Convert the 0-1 observation into state numbers (1,2,3)
            if s == 0.0:
                state = 1
            elif s == 1.0:
                state = 3
            else:
                state = 2
            # covert the 1/2/3 observation into 0/1/2 index
            obs_idx = state - 1
            # apply Bayesian update with deterministic observation model
            pi_new = np.zeros_like(pi)
            pi_new[obs_idx] = 1.0
            # update transition parameters based on the new observation
            # if self.sampled_times[idx_str] == 0 and self.args.tasksampler.hmm_sample_init_pi:
            #     alpha_new = self.update_transition_params(alpha, pi_before, pi_new, learning_rate=0.0)
            alpha_new = self.update_transition_params(alpha, pi_before, pi_new)
            # predict next state distribution based on the updated transition parameters
            pi_next_prior = self.predict_next_state(alpha_new, pi_new)
            pi_next_prior /= pi_next_prior.sum()  # ensure normalization
            # update the stored parameters
            self.alphas[idx_str] = alpha_new
            self.pis_before[idx_str] = pi_new
            self.pis[idx_str] = pi_next_prior
            self.sampled_times[idx_str] += 1
        # For idx not in indices, self.alpha and self.pis are updated as follows
        str_indices = [str(i) for i in indices]
        for idx in self.alphas.keys():
            if idx not in str_indices:
                idx_str = str(idx)
                alpha, pi, pi_before = self.get_alpha_pi(idx_str)
                # without new observation, the Bayesian update defaults to the prior
                pi_new = pi
                # decay the transition parameters without new observations
                alpha_new = self.update_transition_params(alpha, pi_before, pi_new, learning_rate=0.0)
                # predict next state distribution based on the updated transition parameters
                pi_next_prior = self.predict_next_state(alpha_new, pi_new)
                pi_next_prior /= pi_next_prior.sum()  # ensure normalization
                # update the stored parameters
                self.alphas[idx_str] = alpha_new
                self.pis_before[idx_str] = pi_new
                self.pis[idx_str] = pi_next_prior
        return None, None, None

    def compute_statistics(self, metrics):
        """
        Compute statistics of the transition parameters and state distributions for monitoring and analysis.
        """
        alpha_values = np.array(list(self.alphas.values()))
        pi_values = np.array(list(self.pis.values()))
        metrics['tasksample_hmm/state_distribution_statelevel_entropy'] = -np.sum(pi_values * np.log(pi_values + 1e-10), axis=1).mean()
        metrics['tasksample_hmm/transition_matrix_datalevel_variance'] = np.var(alpha_values, axis=0).mean()
        metrics['tasksample_hmm/state_distribution_datalevel_variance'] = np.var(pi_values, axis=0).mean()
        metrics['tasksample_hmm/state_distribution_statelevel_max'] = pi_values.max(axis=1).mean()
        metrics['tasksample_hmm/state_distribution_statelevel_min'] = pi_values.min(axis=1).mean()
        # metrics['tasksample_hmm/transition_matrix_all_max'] = alpha_values.max()
        # metrics['tasksample_hmm/transition_matrix_all_min'] = alpha_values.min()
        # metrics['tasksample_hmm/state_distribution_3_mean'] = pi_values[:,2].mean()
        # metrics['tasksample_hmm/state_distribution_2_mean'] = pi_values[:,1].mean()
        # metrics['tasksample_hmm/state_distribution_1_mean'] = pi_values[:,0].mean()
        # metrics['tasksample_hmm/sample_times_mean'] = np.mean(list(self.sampled_times.values()))
        metrics['tasksample_hmm/sample_times_std'] = np.std(list(self.sampled_times.values()))
        metrics['tasksample_hmm/sample_times_max'] = np.max(list(self.sampled_times.values()))
        metrics['tasksample_hmm/sample_times_min'] = np.min(list(self.sampled_times.values()))
        return metrics

    def save(self, save_path):
        """
        save the hmm model parameters to a JSON file
        """
        import json
        
        data = {
            "alphas": {idx: alpha.tolist() for idx, alpha in self.alphas.items()},
            "pis": {idx: pi.tolist() for idx, pi in self.pis.items()},
            "pis_before": {idx: pi.tolist() for idx, pi in self.pis_before.items()},
            "num_states": self.num_states,
            "sampled_times": self.sampled_times,
            "prior_alpha": self.prior_alpha.tolist()
        }
        with open(os.path.join(save_path, 'hmm_params.json'), 'w') as f:
            json.dump(data, f)

    def load(self, load_path):
        """
        load the hmm model parameters from a JSON file
        """
        try:
            import json
            
            with open(os.path.join(load_path, 'hmm_params.json'), 'r') as f:
                data = json.load(f)
                
            self.alphas = {idx: np.array(alpha) for idx, alpha in data["alphas"].items()}
            self.pis = {idx: np.array(pi) for idx, pi in data["pis"].items()}
            self.pis_before = {idx: np.array(pi) for idx, pi in data["pis_before"].items()}
            self.num_states = data["num_states"]
            self.sampled_times = data["sampled_times"]
            self.prior_alpha = np.array(data["prior_alpha"])
            print(f'Successfully loaded Bayesian HMM from {load_path}')
        except Exception as e:
            print(f'Failed to load Bayesian HMM from {load_path}: {e}')
import numpy as np
from scipy.stats import multivariate_normal
import copy
import tqdm

class fuzzing:
    def __init__(self):
        self.corpus = []
        self.rewards = []
        self.result = []
        self.entropy = []
        self.coverage = []
        self.original = []
        self.count = []
        self.state_cvg = []

        self.sequences = []
        self.current_pose = None
        self.current_reward = None
        self.current_entropy = None
        self.current_coverage = None
        self.current_original = None
        self.current_index = None
        self.current_envsetting = None

        self.GMM = None
        self.GMMupdate = None
        self.GMMK = 1

        self.GMM_cond = None
        self.GMMupdate_cond = None
        self.GMMK_cond = 1
        self.GMMthreshold = 0.01

    def get_pose(self):
        choose_index = np.random.choice(range(len(self.corpus)), 1, p=self.entropy / np.array(self.entropy).sum())[0]
        self.count[choose_index] -= 1
        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_coverage = self.coverage[choose_index]
        self.current_original = self.original[choose_index]
        if self.count[choose_index] <= 0:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None

        return self.current_pose

    def add_crash(self, result_pose):
        self.result.append(result_pose)
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None

    def further_mutation(self, current_pose, rewards, entropy, cvg, original):
        choose_index = self.current_index
        copy_pose = copy.deepcopy(current_pose)
        if choose_index != None:
            self.corpus[choose_index] = copy_pose
            self.rewards[choose_index] = rewards
            self.entropy[choose_index] = entropy
            self.coverage[choose_index] = cvg
            self.count[choose_index] = 20
        else:
            self.corpus.append(copy_pose)
            self.rewards.append(rewards)
            self.entropy.append(entropy)
            self.coverage.append(cvg)
            self.original.append(original)
            self.count.append(20)

    def mutate(self, orig_pos):
        new_pos = copy.deepcopy(orig_pos)
        for i in range(len(orig_pos)):
            new_pos[i] = new_pos[i] + np.random.uniform(-0.05, 0.05, new_pos[i].shape)
            new_pos[i] = np.maximum(-1, new_pos[i])
            new_pos[i] = np.minimum(1, new_pos[i])
        self.current_pose = new_pos
        return new_pos

    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None

    def flatten_states(self, states):
        states = np.array(states)
        states_cond = np.zeros((states.shape[0]-1, states.shape[1] * 2))
        for i in range(states.shape[0]-1):
            states_cond[i] = np.hstack((states[i], states[i + 1]))

        return states, states_cond

    def GMMinit(self, data_corpus, data_corpus_cond):
        res = []
        for i in range(self.GMMK):
            temp = dict()
            temp[0] = 1 / self.GMMK
            temp[1] = temp[0] * np.mean(data_corpus[i:i+15], axis=0)
            temp[2] = np.zeros((data_corpus.shape[1], data_corpus.shape[1]))
            for j in range(i, i+15):
                temp[2] += temp[0] * np.matmul(data_corpus[j: j+1].T, data_corpus[j: j+1])
            temp[2] /= 15
            res.append(temp)

        weights = np.zeros(self.GMMK)
        means = np.zeros((self.GMMK, data_corpus.shape[1]))
        covariances = np.zeros((self.GMMK, data_corpus.shape[1], data_corpus.shape[1]))
        for i in range(self.GMMK):
            weights[i] = res[i][0]
            means[i] = res[i][1] / res[i][0]
            covariances[i] = np.eye(data_corpus.shape[1])

        self.GMM = dict()
        self.GMM['means'] = copy.deepcopy(means)
        self.GMM['weights'] = copy.deepcopy(weights)
        self.GMM['covariances'] = copy.deepcopy(covariances)

        res_cond = []
        for i in range(self.GMMK_cond):
            temp_cond = dict()
            temp_cond[0] = 1 / self.GMMK_cond
            temp_cond[1] = temp_cond[0] * np.mean(data_corpus_cond[i:i+15], axis=0)
            temp_cond[2] = np.zeros((data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
            for j in range(i, i+15):
                temp_cond[2] += temp_cond[0] * np.matmul(data_corpus_cond[j: j+1].T, data_corpus_cond[j: j+1])
            temp_cond[2] /= 15
            res_cond.append(temp_cond)

        weights_cond = np.zeros(self.GMMK_cond)
        means_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1]))
        covariances_cond = np.zeros((self.GMMK_cond, data_corpus_cond.shape[1], data_corpus_cond.shape[1]))
        for i in range(self.GMMK_cond):
            weights_cond[i] = res_cond[i][0]
            means_cond[i] = res_cond[i][1] / res_cond[i][0]
            covariances_cond[i] = np.eye(data_corpus_cond.shape[1])

        self.GMM_cond = dict()
        self.GMM_cond['means'] = copy.deepcopy(means_cond)
        self.GMM_cond['weights'] = copy.deepcopy(weights_cond)
        self.GMM_cond['covariances'] = copy.deepcopy(covariances_cond)

        return res, res_cond

    def get_mdp_pdf(self, states_seq, states_seq_cond):
        first_frame = states_seq[0:1]
        GMMpdf = np.zeros(self.GMMK)
        for k in range(self.GMMK):
            GMMpdf[k] = self.GMM['weights'][k] * multivariate_normal.pdf(first_frame, self.GMM['means'][k], self.GMM['covariances'][k])
        GMMpdf += 1e-5
        GMMpdfvalue = np.sum(GMMpdf)
        first_frame_pdf = GMMpdf

        single_frame_pdf = np.zeros((states_seq.shape[0], self.GMMK))
        other_frame_pdf = np.zeros((states_seq_cond.shape[0], self.GMMK_cond))

        for i in range(states_seq.shape[0]):
            for k in range(self.GMMK):
                single_frame_pdf[i, k] = self.GMM['weights'][k] * multivariate_normal.pdf(states_seq[i], self.GMM['means'][k], self.GMM['covariances'][k])
        single_frame_pdf += 1e-5

        for i in range(states_seq_cond.shape[0]):
            for k in range(self.GMMK_cond):
                other_frame_pdf[i, k] = self.GMM_cond['weights'][k] * multivariate_normal.pdf(states_seq_cond[i], self.GMM_cond['means'][k], self.GMM_cond['covariances'][k])
            other_frame_pdf[i] += 1e-5
            GMMpdfvalue *= np.min([np.sum(other_frame_pdf[i]) / np.sum(single_frame_pdf[i]), 1.0])
        return GMMpdfvalue, GMMpdf, other_frame_pdf

    def state_coverage(self, states_seq):
        states_seq, states_seq_cond = self.flatten_states(states_seq)
        if self.GMM == None:
            GMMresult, GMMresult_cond = self.GMMinit(states_seq, states_seq_cond)
            self.GMMupdate = dict()
            self.GMMupdate_cond = dict()
            self.GMMupdate['iter'] = 10
            self.GMMupdate['threshold'] = 0.05
            self.GMMupdate['S'] = copy.deepcopy(GMMresult)
            self.GMMupdate_cond['S'] = copy.deepcopy(GMMresult_cond)

        GMMpdfvalue, GMMpdf, other_frame_pdf = self.get_mdp_pdf(states_seq, states_seq_cond)
        first_frame = states_seq[0:1, :]

        if GMMpdfvalue < self.GMMthreshold:
            gamma = 1.0 / (self.GMMupdate['iter'])
            GMMpdf /= np.sum(GMMpdf)
            new_S = copy.deepcopy(self.GMMupdate['S'])

            for i in range(self.GMMK):
                new_S[i][0] = self.GMMupdate['S'][i][0] + gamma * (GMMpdf[i] - self.GMMupdate['S'][i][0])
                new_S[i][1] = self.GMMupdate['S'][i][1] + gamma * (GMMpdf[i]*first_frame - self.GMMupdate['S'][i][1])
                new_S[i][2] = self.GMMupdate['S'][i][2] + gamma * (GMMpdf[i]*np.matmul(first_frame.T, first_frame) - self.GMMupdate['S'][i][2])

            self.GMMupdate['S'] = copy.deepcopy(new_S)

            for i in range(self.GMMK):
                self.GMM['weights'][i] = new_S[i][0]
                self.GMM['means'][i] = new_S[i][1] / new_S[i][0]
                self.GMM['covariances'][i] = (new_S[i][2] - np.matmul(self.GMM['means'][i].reshape(1, -1).T, new_S[i][1])) / new_S[i][0]
                W, V = np.linalg.eigh(self.GMM['covariances'][i])
                W = np.maximum(W, 1e-3)
                D = np.diag(W)
                reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                self.GMM['covariances'][i] = copy.deepcopy(reconstruction)

            cond_choices = np.argsort(np.sum(other_frame_pdf, axis=1))
            for cond_index in cond_choices[:cond_choices.shape[0] // 10]:
                GMMpdf_cond = other_frame_pdf[cond_index]
                GMMpdf_cond /= np.sum(GMMpdf_cond)
                current_frame = states_seq_cond[cond_index: cond_index + 1, :]
                new_S_cond = copy.deepcopy(self.GMMupdate_cond['S'])

                for i in range(self.GMMK_cond):
                    new_S_cond[i][0] = self.GMMupdate_cond['S'][i][0] + gamma * (GMMpdf_cond[i] - self.GMMupdate_cond['S'][i][0])
                    new_S_cond[i][1] = self.GMMupdate_cond['S'][i][1] + gamma * (GMMpdf_cond[i]*current_frame - self.GMMupdate_cond['S'][i][1])
                    new_S_cond[i][2] = self.GMMupdate_cond['S'][i][2] + gamma * (GMMpdf_cond[i]*np.matmul(current_frame.T, current_frame) - self.GMMupdate_cond['S'][i][2])

                self.GMMupdate_cond['S'] = copy.deepcopy(new_S_cond)

                for i in range(self.GMMK_cond):
                    self.GMM_cond['weights'][i] = new_S_cond[i][0]
                    self.GMM_cond['means'][i] = new_S_cond[i][1] / new_S_cond[i][0]
                    self.GMM_cond['covariances'][i] = (new_S_cond[i][2] - np.matmul(self.GMM_cond['means'][i].reshape(1, -1).T, new_S_cond[i][1])) / new_S_cond[i][0]
                    W, V = np.linalg.eigh(self.GMM_cond['covariances'][i])
                    W = np.maximum(W, 1e-3)
                    D = np.diag(W)
                    reconstruction = np.matmul(np.matmul(V, D), np.linalg.inv(V))
                    self.GMM_cond['covariances'][i] = copy.deepcopy(reconstruction)

        return GMMpdfvalue

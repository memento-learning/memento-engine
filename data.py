from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import pickle
import os.path
from os import path

class DuoLingo(Dataset):
    """Duo Lingo user log traces"""

    def __init__(self):
        # Build feather dataset for faster loading
        if not path.exists('./data/duolingo_trace.feather'):
            temp_df = pd.read_csv('./data/settles.acl16.learning_traces.13m.csv')
            temp_df.to_feather('./data/duolingo_trace.feather')

        df = pd.read_feather('./data/duolingo_trace.feather')
        # subset data frame
        df_sub = df[["p_recall", "timestamp","delta", "user_id", "lexeme_id", "history_seen", "history_correct", "session_seen", "session_correct"]]
        raw_data = df_sub.values

        if path.exists('./data/duolingo_user_item_ctx.pkl'):
            with open('./data/duolingo_user_item_ctx.pkl', 'rb') as f:
               self.user_item_ctx = pickle.load(f)
            with open('./data/duolingo_item_ctx.pkl', 'rb') as f:
               self.item_ctx = pickle.load(f)  
        else:
            user_item_ctx = {}
            item_ctx = {}
            # build user, and user_item ctx
            for i, row in enumerate(raw_data):
                user_id = row[3]
                item_id = row[4]
                
                if user_id not in user_item_ctx:
                    user_item_ctx[user_id] = {}
                    
                if item_id not in user_item_ctx[user_id]:
                    user_item_ctx[user_id][item_id] = []
                    
                if item_id not in item_ctx:
                    item_ctx[item_id] = []

                user_item_ctx[user_id][item_id].append(i)
                item_ctx[item_id].append(i)

            with open('./data/duolingo_user_item_ctx.pkl', 'wb') as f:
               pickle.dump(user_item_ctx, f, pickle.HIGHEST_PROTOCOL)
            with open('./data/duolingo_item_ctx.pkl', 'wb') as f:
               pickle.dump(item_ctx, f, pickle.HIGHEST_PROTOCOL) 

            self.user_item_ctx = user_item_ctx
            self.item_ctx = item_ctx  
                

        # TODO:  Extract and store features

        # for i, row in enumerate(raw_data):
        #     pass
    
        self.raw_data = raw_data
        self.MAX_SESSIONS = 400


    def __len__(self):
       return len(self.raw_data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        delta = self.raw_data[idx, 2] / 86400.0 # Normalize to days
        p = self.raw_data[idx, 0] * 1.0
        correct = self.raw_data[idx, 6]
        incorrect = self.raw_data[idx, 5] - correct

        user_id = self.raw_data[idx, 3]
        item_id = self.raw_data[idx, 4]

        hist_p = []
        
        if isinstance(idx, list):
            for i, uid in enumerate(user_id):
                all_sessions = self.user_item_ctx[uid][item_id[i]]
                prev_sessions = all_sessions[:all_sessions.index(idx[i])+1]
                hist_p_elem = self.raw_data[prev_sessions[:-1], 0]
                hist_p_elem = np.concatenate(([self.raw_data[prev_sessions[0]][6] * 1.0 / self.raw_data[prev_sessions[0]][5]], hist_p_elem))
                hist_p_elem = np.pad(np.array(hist_p_elem), (0, self.MAX_SESSIONS - len(hist_p_elem)), 'constant', constant_values=-1)
                hist_p.append(hist_p_elem)
        else:
            all_sessions = self.user_item_ctx[user_id][item_id]
            prev_sessions = all_sessions[:all_sessions.index(idx)+1]
            hist_p = self.raw_data[prev_sessions[:-1], 0]
            hist_p = np.concatenate(([self.raw_data[prev_sessions[0]][6] * 1.0 / self.raw_data[prev_sessions[0]][5]], hist_p))
            hist_p = np.pad(hist_p, (0, self.MAX_SESSIONS - len(hist_p)), 'constant', constant_values=-1)

        hist_p = np.array(hist_p).astype(np.float32)
        sample = {
            'delta': delta, 
            'p': p,
            'correct': correct,
            'incorrect': incorrect,
            'hist_p': hist_p
        }

        return sample


if __name__ == "__main__":
    dataset = DuoLingo()
    print(dataset[0])
    print(dataset[[100, 108]])
        

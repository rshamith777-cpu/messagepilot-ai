import os
import pandas as pd
from typing import Dict, Any

class DataLoader:
    """Loads and indexes all dataset CSV files for rapid lookup."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.messages_df = None
        self.users_df = None
        self.groups_df = None
        self.group_members_df = None
        self.business_accounts_df = None
        self.user_business_history_df = None
        self.message_history_df = None
        self.message_events_df = None
        self.images_df = None
        self.voice_notes_df = None
        self.daily_notification_summary_df = None
        self.sample_messages_df = None
        
        # Indexed maps for fast lookups
        self.users_map: Dict[str, Dict[str, Any]] = {}
        self.groups_map: Dict[str, Dict[str, Any]] = {}
        self.group_members_map: Dict[str, Dict[str, Dict[str, Any]]] = {} # (group_id, user_id) -> record
        self.business_map: Dict[str, Dict[str, Any]] = {}
        self.user_business_history_map: Dict[str, Dict[str, Dict[str, Any]]] = {} # (user_id, business_id) -> record
        self.images_map: Dict[str, str] = {}
        self.voice_notes_map: Dict[str, str] = {}

    def load_all(self):
        """Loads all CSVs and populates lookup indices."""
        def load_csv(filename: str) -> pd.DataFrame:
            path = os.path.join(self.dataset_dir, filename)
            if os.path.exists(path):
                return pd.read_csv(path).fillna("")
            return pd.DataFrame()

        self.messages_df = load_csv("messages.csv")
        self.users_df = load_csv("users.csv")
        self.groups_df = load_csv("groups.csv")
        self.group_members_df = load_csv("group_members.csv")
        self.business_accounts_df = load_csv("business_accounts.csv")
        self.user_business_history_df = load_csv("user_business_history.csv")
        self.message_history_df = load_csv("message_history.csv")
        self.message_events_df = load_csv("message_events.csv")
        self.images_df = load_csv("images.csv")
        self.voice_notes_df = load_csv("voice_notes.csv")
        self.daily_notification_summary_df = load_csv("daily_notification_summary.csv")
        self.sample_messages_df = load_csv("sample_messages.csv")
        
        self._build_indices()
        print(f"Loaded {len(self.messages_df)} incoming messages, {len(self.users_df)} users, {len(self.groups_df)} groups, {len(self.business_accounts_df)} businesses.")

    def _build_indices(self):
        # Users index
        for _, row in self.users_df.iterrows():
            self.users_map[str(row['user_id'])] = row.to_dict()

        # Groups index
        for _, row in self.groups_df.iterrows():
            self.groups_map[str(row['group_id'])] = row.to_dict()

        # Group members index
        for _, row in self.group_members_df.iterrows():
            g_id = str(row['group_id'])
            u_id = str(row['user_id'])
            if g_id not in self.group_members_map:
                self.group_members_map[g_id] = {}
            self.group_members_map[g_id][u_id] = row.to_dict()

        # Business accounts index
        for _, row in self.business_accounts_df.iterrows():
            self.business_map[str(row['business_id'])] = row.to_dict()

        # User business history index
        for _, row in self.user_business_history_df.iterrows():
            u_id = str(row['user_id'])
            b_id = str(row['business_id'])
            if u_id not in self.user_business_history_map:
                self.user_business_history_map[u_id] = {}
            self.user_business_history_map[u_id][b_id] = row.to_dict()

        # Media indices
        for _, row in self.images_df.iterrows():
            self.images_map[str(row['image_id'])] = str(row['file_path'])

        for _, row in self.voice_notes_df.iterrows():
            self.voice_notes_map[str(row['voice_note_id'])] = str(row['file_path'])

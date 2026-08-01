import os
import pandas as pd
import logging
from typing import Dict, List, Optional
from models import (
    Message, User, Group, GroupMember, Business, UserBusinessHistory,
    MessageHistory, MessageEvent, Image, VoiceNote, NotificationSummary,
    MessageContext, logger
)

class DataLoader:
    """Reads dataset CSV files, validates schemas, instantiates Dataclasses, and builds MessageContext."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.messages: List[Message] = []
        self.users: Dict[str, User] = {}
        self.groups: Dict[str, Group] = {}
        self.group_members: Dict[str, GroupMember] = {} # Key: "(group_id, user_id)"
        self.businesses: Dict[str, Business] = {}
        self.user_business_histories: Dict[str, UserBusinessHistory] = {} # Key: "(user_id, business_id)"
        self.message_history: List[MessageHistory] = []
        self.message_events: Dict[str, List[MessageEvent]] = {} # Key: user_id
        self.images: Dict[str, Image] = {}
        self.voice_notes: Dict[str, VoiceNote] = {}
        self.notification_summaries: Dict[str, List[NotificationSummary]] = {} # Key: user_id

    def load_all(self):
        """Loads and validates all CSV files into Dataclasses and indexed maps."""
        logger.info(f"Loading datasets from {self.dataset_dir}")
        self._load_messages()
        self._load_users()
        self._load_groups()
        self._load_group_members()
        self._load_businesses()
        self._load_user_business_history()
        self._load_message_history()
        self._load_message_events()
        self._load_images()
        self._load_voice_notes()
        self._load_notification_summaries()
        logger.info("Dataset loading and validation completed successfully.")

    def _read_csv(self, filename: str, required_cols: List[str]) -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file missing: {path}")
        df = pd.read_csv(path).fillna("")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Schema validation error in {filename}: missing columns {missing}")
        return df

    def _load_messages(self):
        cols = ['message_id', 'user_id', 'conversation_type', 'group_id', 'business_id', 'sender_user_id', 'created_at', 'message_text', 'media_type', 'media_id', 'forwarded_count']
        df = self._read_csv('messages.csv', cols)
        for _, r in df.iterrows():
            self.messages.append(Message(
                message_id=str(r['message_id']),
                user_id=str(r['user_id']),
                conversation_type=str(r['conversation_type']),
                group_id=str(r['group_id']),
                business_id=str(r['business_id']),
                sender_user_id=str(r['sender_user_id']),
                created_at=str(r['created_at']),
                message_text=str(r['message_text']),
                media_type=str(r['media_type']),
                media_id=str(r['media_id']),
                forwarded_count=int(r['forwarded_count'] or 0)
            ))
        logger.info(f"Loaded {len(self.messages)} target messages from messages.csv")

    def _load_users(self):
        cols = ['user_id', 'do_not_disturb_window', 'messages_opened_30d', 'messages_replied_30d', 'notifications_dismissed_30d', 'messages_reported_30d']
        df = self._read_csv('users.csv', cols)
        for _, r in df.iterrows():
            u = User(
                user_id=str(r['user_id']),
                do_not_disturb_window=str(r['do_not_disturb_window']),
                messages_opened_30d=int(r['messages_opened_30d'] or 0),
                messages_replied_30d=int(r['messages_replied_30d'] or 0),
                notifications_dismissed_30d=int(r['notifications_dismissed_30d'] or 0),
                messages_reported_30d=int(r['messages_reported_30d'] or 0)
            )
            self.users[u.user_id] = u

    def _load_groups(self):
        cols = ['group_id', 'group_name', 'group_type', 'member_count', 'admin_count', 'created_at', 'messages_30d']
        df = self._read_csv('groups.csv', cols)
        for _, r in df.iterrows():
            g = Group(
                group_id=str(r['group_id']),
                group_name=str(r['group_name']),
                group_type=str(r['group_type']),
                member_count=int(r['member_count'] or 0),
                admin_count=int(r['admin_count'] or 0),
                created_at=str(r['created_at']),
                messages_30d=int(r['messages_30d'] or 0)
            )
            self.groups[g.group_id] = g

    def _load_group_members(self):
        cols = ['group_id', 'user_id', 'role', 'joined_at', 'messages_sent_30d', 'messages_read_30d', 'replies_sent_30d', 'notifications_dismissed_30d', 'group_muted_by_user']
        df = self._read_csv('group_members.csv', cols)
        for _, r in df.iterrows():
            gm = GroupMember(
                group_id=str(r['group_id']),
                user_id=str(r['user_id']),
                role=str(r['role']),
                joined_at=str(r['joined_at']),
                messages_sent_30d=int(r['messages_sent_30d'] or 0),
                messages_read_30d=int(r['messages_read_30d'] or 0),
                replies_sent_30d=int(r['replies_sent_30d'] or 0),
                notifications_dismissed_30d=int(r['notifications_dismissed_30d'] or 0),
                group_muted_by_user=int(r['group_muted_by_user'] or 0)
            )
            self.group_members[f"({gm.group_id},{gm.user_id})"] = gm

    def _load_businesses(self):
        cols = ['business_id', 'display_name', 'brand_name', 'category', 'verified', 'official_domain', 'domain_used_by_sender', 'account_age_days', 'messages_sent_30d', 'user_reports_30d', 'domain_used_by_sender_age_days']
        df = self._read_csv('business_accounts.csv', cols)
        for _, r in df.iterrows():
            b = Business(
                business_id=str(r['business_id']),
                display_name=str(r['display_name']),
                brand_name=str(r['brand_name']),
                category=str(r['category']),
                verified=int(r['verified'] or 0),
                official_domain=str(r['official_domain']),
                domain_used_by_sender=str(r['domain_used_by_sender']),
                account_age_days=int(r['account_age_days'] or 0),
                messages_sent_30d=int(r['messages_sent_30d'] or 0),
                user_reports_30d=int(r['user_reports_30d'] or 0),
                domain_used_by_sender_age_days=int(r['domain_used_by_sender_age_days'] or 0)
            )
            self.businesses[b.business_id] = b

    def _load_user_business_history(self):
        cols = ['user_id', 'business_id', 'why_user_knows_account', 'last_activity_at', 'allows_promotions', 'promotions_opted_out_at', 'activity_count_180d', 'messages_opened_30d', 'messages_dismissed_30d', 'messages_replied_30d', 'last_reply_at']
        df = self._read_csv('user_business_history.csv', cols)
        for _, r in df.iterrows():
            ubh = UserBusinessHistory(
                user_id=str(r['user_id']),
                business_id=str(r['business_id']),
                why_user_knows_account=str(r['why_user_knows_account']),
                last_activity_at=str(r['last_activity_at']),
                allows_promotions=int(r['allows_promotions'] if r['allows_promotions'] != "" else 1),
                promotions_opted_out_at=str(r['promotions_opted_out_at']),
                activity_count_180d=int(r['activity_count_180d'] or 0),
                messages_opened_30d=int(r['messages_opened_30d'] or 0),
                messages_dismissed_30d=int(r['messages_dismissed_30d'] or 0),
                messages_replied_30d=int(r['messages_replied_30d'] or 0),
                last_reply_at=str(r['last_reply_at'])
            )
            self.user_business_histories[f"({ubh.user_id},{ubh.business_id})"] = ubh

    def _load_message_history(self):
        cols = ['message_id', 'user_id', 'conversation_type', 'group_id', 'business_id', 'sender_user_id', 'created_at', 'message_text', 'media_type', 'media_id', 'forwarded_count']
        df = self._read_csv('message_history.csv', cols)
        for _, r in df.iterrows():
            self.message_history.append(MessageHistory(
                message_id=str(r['message_id']),
                user_id=str(r['user_id']),
                conversation_type=str(r['conversation_type']),
                group_id=str(r['group_id']),
                business_id=str(r['business_id']),
                sender_user_id=str(r['sender_user_id']),
                created_at=str(r['created_at']),
                message_text=str(r['message_text']),
                media_type=str(r['media_type']),
                media_id=str(r['media_id']),
                forwarded_count=int(r['forwarded_count'] or 0)
            ))

    def _load_message_events(self):
        cols = ['user_id', 'message_id', 'message_opened', 'message_replied', 'reaction_time_minutes', 'notification_dismissed', 'muted_after_message', 'message_reported']
        df = self._read_csv('message_events.csv', cols)
        for _, r in df.iterrows():
            ev = MessageEvent(
                user_id=str(r['user_id']),
                message_id=str(r['message_id']),
                message_opened=int(r['message_opened'] or 0),
                message_replied=int(r['message_replied'] or 0),
                reaction_time_minutes=float(r['reaction_time_minutes'] or 0.0),
                notification_dismissed=int(r['notification_dismissed'] or 0),
                muted_after_message=int(r['muted_after_message'] or 0),
                message_reported=int(r['message_reported'] or 0)
            )
            if ev.user_id not in self.message_events:
                self.message_events[ev.user_id] = []
            self.message_events[ev.user_id].append(ev)

    def _load_images(self):
        cols = ['image_id', 'file_path']
        df = self._read_csv('images.csv', cols)
        for _, r in df.iterrows():
            img = Image(image_id=str(r['image_id']), file_path=str(r['file_path']))
            self.images[img.image_id] = img

    def _load_voice_notes(self):
        cols = ['voice_note_id', 'file_path']
        df = self._read_csv('voice_notes.csv', cols)
        for _, r in df.iterrows():
            vn = VoiceNote(voice_note_id=str(r['voice_note_id']), file_path=str(r['file_path']))
            self.voice_notes[vn.voice_note_id] = vn

    def _load_notification_summaries(self):
        cols = ['user_id', 'date', 'notifications_sent', 'notifications_dismissed']
        df = self._read_csv('daily_notification_summary.csv', cols)
        for _, r in df.iterrows():
            ns = NotificationSummary(
                user_id=str(r['user_id']),
                date=str(r['date']),
                notifications_sent=int(r['notifications_sent'] or 0),
                notifications_dismissed=int(r['notifications_dismissed'] or 0)
            )
            if ns.user_id not in self.notification_summaries:
                self.notification_summaries[ns.user_id] = []
            self.notification_summaries[ns.user_id].append(ns)

    def build_message_context(self, msg: Message) -> MessageContext:
        """Constructs a complete MessageContext object for a given Message."""
        receiver = self.users.get(msg.user_id)
        sender = self.users.get(msg.sender_user_id) if msg.sender_user_id else None
        group = self.groups.get(msg.group_id) if msg.group_id else None
        group_member = self.group_members.get(f"({msg.group_id},{msg.user_id})") if msg.group_id else None
        business = self.businesses.get(msg.business_id) if msg.business_id else None
        business_history = self.user_business_histories.get(f"({msg.user_id},{msg.business_id})") if msg.business_id else None
        
        # User relevant message history
        user_msg_hist = [mh for mh in self.message_history if mh.user_id == msg.user_id]
        user_events = self.message_events.get(msg.user_id, [])
        user_summaries = self.notification_summaries.get(msg.user_id, [])

        img_info = self.images.get(msg.media_id) if msg.media_type == 'image' else None
        vn_info = self.voice_notes.get(msg.media_id) if msg.media_type == 'voice' else None

        return MessageContext(
            current_message=msg,
            receiver=receiver,
            sender=sender,
            group=group,
            group_member=group_member,
            business=business,
            business_history=business_history,
            message_history=user_msg_hist,
            message_events=user_events,
            notification_summary=user_summaries,
            image_info=img_info,
            voice_note_info=vn_info
        )

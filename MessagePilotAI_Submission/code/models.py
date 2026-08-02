import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MessageRouter")

@dataclass
class Message:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: Optional[str] = ""
    business_id: Optional[str] = ""
    sender_user_id: Optional[str] = ""
    created_at: str = ""
    message_text: Optional[str] = ""
    media_type: Optional[str] = ""
    media_id: Optional[str] = ""
    forwarded_count: int = 0

@dataclass
class User:
    user_id: str
    do_not_disturb_window: str = ""
    messages_opened_30d: int = 0
    messages_replied_30d: int = 0
    notifications_dismissed_30d: int = 0
    messages_reported_30d: int = 0

@dataclass
class Group:
    group_id: str
    group_name: str = ""
    group_type: str = ""
    member_count: int = 0
    admin_count: int = 0
    created_at: str = ""
    messages_30d: int = 0

@dataclass
class GroupMember:
    group_id: str
    user_id: str
    role: str = ""
    joined_at: str = ""
    messages_sent_30d: int = 0
    messages_read_30d: int = 0
    replies_sent_30d: int = 0
    notifications_dismissed_30d: int = 0
    group_muted_by_user: int = 0

@dataclass
class Business:
    business_id: str
    display_name: str = ""
    brand_name: str = ""
    category: str = ""
    verified: int = 0
    official_domain: str = ""
    domain_used_by_sender: str = ""
    account_age_days: int = 0
    messages_sent_30d: int = 0
    user_reports_30d: int = 0
    domain_used_by_sender_age_days: int = 0

@dataclass
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str = ""
    last_activity_at: str = ""
    allows_promotions: int = 1
    promotions_opted_out_at: str = ""
    activity_count_180d: int = 0
    messages_opened_30d: int = 0
    messages_dismissed_30d: int = 0
    messages_replied_30d: int = 0
    last_reply_at: str = ""

@dataclass
class MessageHistory:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str = ""
    business_id: str = ""
    sender_user_id: str = ""
    created_at: str = ""
    message_text: str = ""
    media_type: str = ""
    media_id: str = ""
    forwarded_count: int = 0

@dataclass
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: int = 0
    message_replied: int = 0
    reaction_time_minutes: float = 0.0
    notification_dismissed: int = 0
    muted_after_message: int = 0
    message_reported: int = 0

@dataclass
class Image:
    image_id: str
    file_path: str

@dataclass
class VoiceNote:
    voice_note_id: str
    file_path: str

@dataclass
class NotificationSummary:
    user_id: str
    date: str
    notifications_sent: int = 0
    notifications_dismissed: int = 0

@dataclass
class MessageContext:
    current_message: Message
    receiver: Optional[User] = None
    sender: Optional[User] = None
    group: Optional[Group] = None
    group_member: Optional[GroupMember] = None
    business: Optional[Business] = None
    business_history: Optional[UserBusinessHistory] = None
    message_history: List[MessageHistory] = field(default_factory=list)
    message_events: List[MessageEvent] = field(default_factory=list)
    notification_summary: List[NotificationSummary] = field(default_factory=list)
    image_info: Optional[Image] = None
    voice_note_info: Optional[VoiceNote] = None

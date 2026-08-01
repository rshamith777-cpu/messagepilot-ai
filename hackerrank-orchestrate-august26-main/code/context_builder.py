from typing import Dict, Any
from data_loader import DataLoader

class ContextBuilder:
    """Builds an enriched context object for a specific incoming message."""

    def __init__(self, data_loader: DataLoader):
        self.dl = data_loader

    def build_context(self, message_row: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(message_row.get('user_id', ''))
        group_id = str(message_row.get('group_id', ''))
        business_id = str(message_row.get('business_id', ''))
        sender_id = str(message_row.get('sender_user_id', ''))
        
        user_info = self.dl.users_map.get(user_id, {})
        group_info = self.dl.groups_map.get(group_id, {})
        group_member_info = self.dl.group_members_map.get(group_id, {}).get(user_id, {})
        sender_group_member_info = self.dl.group_members_map.get(group_id, {}).get(sender_id, {})
        
        business_info = self.dl.business_map.get(business_id, {})
        user_business_info = self.dl.user_business_history_map.get(user_id, {}).get(business_id, {})

        media_type = str(message_row.get('media_type', ''))
        media_id = str(message_row.get('media_id', ''))
        media_path = ""
        if media_type == 'image':
            media_path = self.dl.images_map.get(media_id, "")
        elif media_type == 'voice':
            media_path = self.dl.voice_notes_map.get(media_id, "")

        context = {
            'message': message_row,
            'user': user_info,
            'group': group_info,
            'group_member': group_member_info,
            'sender_group_member': sender_group_member_info,
            'business': business_info,
            'user_business': user_business_info,
            'media_path': media_path
        }
        return context

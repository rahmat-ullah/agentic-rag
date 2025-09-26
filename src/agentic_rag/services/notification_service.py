"""
Notification Service

This module provides notification capabilities for alerts and critical issues,
including email, Slack, webhook, and PagerDuty integrations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
from enum import Enum

import aiohttp
import structlog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from agentic_rag.config import get_settings
from agentic_rag.models.monitoring import Alert, AlertSeverity, AlertStatus

logger = structlog.get_logger(__name__)


class NotificationChannel(str, Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS = "sms"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationTemplate:
    """Notification message template."""
    
    def __init__(
        self,
        subject_template: str,
        body_template: str,
        channel: NotificationChannel
    ):
        self.subject_template = subject_template
        self.body_template = body_template
        self.channel = channel
    
    def format_message(self, alert: Alert, context: Dict[str, Any] = None) -> Dict[str, str]:
        """Format notification message from alert and context."""
        context = context or {}
        
        # Build template context
        template_context = {
            "alert_name": alert.alert_name,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "description": alert.description,
            "service": alert.labels.get("service", "unknown"),
            "environment": alert.labels.get("environment", "unknown"),
            "timestamp": alert.created_at.isoformat(),
            "runbook_url": alert.runbook_url or "N/A",
            "dashboard_url": alert.dashboard_url or "N/A",
            **context
        }
        
        # Format subject and body
        subject = self.subject_template.format(**template_context)
        body = self.body_template.format(**template_context)
        
        return {
            "subject": subject,
            "body": body
        }


class NotificationService:
    """Service for sending notifications through various channels."""
    
    def __init__(self):
        self.settings = get_settings()
        self._session = None
        self._templates = self._load_templates()
        self._escalation_rules = self._load_escalation_rules()
        
        logger.info("Notification service initialized")
    
    async def start(self):
        """Start the notification service."""
        self._session = aiohttp.ClientSession()
        logger.info("Notification service started")
    
    async def stop(self):
        """Stop the notification service."""
        if self._session:
            await self._session.close()
        logger.info("Notification service stopped")
    
    async def send_alert_notification(
        self,
        alert: Alert,
        channels: List[NotificationChannel] = None,
        context: Dict[str, Any] = None
    ):
        """Send notification for an alert."""
        try:
            # Determine channels based on alert severity if not specified
            if not channels:
                channels = self._get_channels_for_severity(alert.severity)
            
            # Send notifications to each channel
            tasks = []
            for channel in channels:
                task = self._send_notification(alert, channel, context)
                tasks.append(task)
            
            # Execute all notifications concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count
            
            logger.info(
                "Alert notification sent",
                alert_id=str(alert.id),
                channels=len(channels),
                success=success_count,
                errors=error_count
            )
            
            return {
                "success": success_count,
                "errors": error_count,
                "total": len(channels)
            }
            
        except Exception as e:
            logger.error("Failed to send alert notification", error=str(e))
            raise
    
    async def send_escalation_notification(
        self,
        alert: Alert,
        escalation_level: int,
        context: Dict[str, Any] = None
    ):
        """Send escalation notification for unresolved alert."""
        try:
            escalation_rule = self._escalation_rules.get(escalation_level)
            if not escalation_rule:
                logger.warning(f"No escalation rule for level {escalation_level}")
                return
            
            # Add escalation context
            escalation_context = {
                "escalation_level": escalation_level,
                "escalation_reason": f"Alert unresolved for {escalation_rule['delay']} minutes",
                **(context or {})
            }
            
            # Send to escalation channels
            await self.send_alert_notification(
                alert,
                escalation_rule["channels"],
                escalation_context
            )
            
            logger.info(
                "Escalation notification sent",
                alert_id=str(alert.id),
                level=escalation_level
            )
            
        except Exception as e:
            logger.error("Failed to send escalation notification", error=str(e))
            raise
    
    async def _send_notification(
        self,
        alert: Alert,
        channel: NotificationChannel,
        context: Dict[str, Any] = None
    ):
        """Send notification to a specific channel."""
        try:
            # Get template for channel
            template = self._templates.get(channel)
            if not template:
                logger.warning(f"No template for channel {channel}")
                return
            
            # Format message
            message = template.format_message(alert, context)
            
            # Send based on channel type
            if channel == NotificationChannel.EMAIL:
                await self._send_email(message, alert)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack(message, alert)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(message, alert)
            elif channel == NotificationChannel.PAGERDUTY:
                await self._send_pagerduty(message, alert)
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
            
        except Exception as e:
            logger.error(f"Failed to send {channel} notification", error=str(e))
            raise
    
    async def _send_email(self, message: Dict[str, str], alert: Alert):
        """Send email notification."""
        try:
            # Email configuration
            smtp_server = self.settings.smtp_server or "localhost"
            smtp_port = self.settings.smtp_port or 587
            smtp_username = self.settings.smtp_username
            smtp_password = self.settings.smtp_password
            from_email = self.settings.smtp_from_email or "alerts@agentic-rag.com"
            
            # Determine recipients based on alert labels
            to_emails = self._get_email_recipients(alert)
            if not to_emails:
                logger.warning("No email recipients configured for alert")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = message["subject"]
            
            # Add body
            msg.attach(MIMEText(message["body"], 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_username and smtp_password:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                
                server.send_message(msg)
            
            logger.info("Email notification sent", recipients=len(to_emails))
            
        except Exception as e:
            logger.error("Failed to send email notification", error=str(e))
            raise
    
    async def _send_slack(self, message: Dict[str, str], alert: Alert):
        """Send Slack notification."""
        try:
            webhook_url = self.settings.slack_webhook_url
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "good"
            }
            color = color_map.get(alert.severity, "warning")
            
            # Create Slack payload
            payload = {
                "text": message["subject"],
                "attachments": [
                    {
                        "color": color,
                        "title": alert.alert_name,
                        "text": message["body"],
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value,
                                "short": True
                            },
                            {
                                "title": "Service",
                                "value": alert.labels.get("service", "unknown"),
                                "short": True
                            }
                        ],
                        "footer": "Agentic RAG Monitoring",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            # Add runbook link if available
            if alert.runbook_url:
                payload["attachments"][0]["actions"] = [
                    {
                        "type": "button",
                        "text": "View Runbook",
                        "url": alert.runbook_url
                    }
                ]
            
            # Send to Slack
            async with self._session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack API returned status {response.status}")
            
            logger.info("Slack notification sent")
            
        except Exception as e:
            logger.error("Failed to send Slack notification", error=str(e))
            raise
    
    async def _send_webhook(self, message: Dict[str, str], alert: Alert):
        """Send webhook notification."""
        try:
            webhook_url = self.settings.webhook_url
            if not webhook_url:
                logger.warning("Webhook URL not configured")
                return
            
            # Create webhook payload
            payload = {
                "alert_id": str(alert.id),
                "alert_name": alert.alert_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "description": alert.description,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "created_at": alert.created_at.isoformat(),
                "message": message
            }
            
            # Send webhook
            async with self._session.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status not in [200, 201, 202]:
                    raise Exception(f"Webhook returned status {response.status}")
            
            logger.info("Webhook notification sent")
            
        except Exception as e:
            logger.error("Failed to send webhook notification", error=str(e))
            raise
    
    async def _send_pagerduty(self, message: Dict[str, str], alert: Alert):
        """Send PagerDuty notification."""
        try:
            integration_key = self.settings.pagerduty_integration_key
            if not integration_key:
                logger.warning("PagerDuty integration key not configured")
                return
            
            # Determine event action based on alert status
            event_action = "trigger"
            if alert.status == AlertStatus.RESOLVED:
                event_action = "resolve"
            elif alert.status == AlertStatus.ACKNOWLEDGED:
                event_action = "acknowledge"
            
            # Create PagerDuty payload
            payload = {
                "routing_key": integration_key,
                "event_action": event_action,
                "dedup_key": f"agentic-rag-{alert.fingerprint}",
                "payload": {
                    "summary": message["subject"],
                    "source": "agentic-rag-monitoring",
                    "severity": alert.severity.value.lower(),
                    "component": alert.labels.get("service", "unknown"),
                    "group": alert.labels.get("team", "platform"),
                    "class": "monitoring",
                    "custom_details": {
                        "description": alert.description,
                        "runbook_url": alert.runbook_url,
                        "dashboard_url": alert.dashboard_url,
                        "labels": alert.labels,
                        "annotations": alert.annotations
                    }
                }
            }
            
            # Send to PagerDuty
            pagerduty_url = "https://events.pagerduty.com/v2/enqueue"
            async with self._session.post(pagerduty_url, json=payload) as response:
                if response.status != 202:
                    raise Exception(f"PagerDuty API returned status {response.status}")
            
            logger.info("PagerDuty notification sent")
            
        except Exception as e:
            logger.error("Failed to send PagerDuty notification", error=str(e))
            raise
    
    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Get notification channels based on alert severity."""
        if severity == AlertSeverity.CRITICAL:
            return [
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK,
                NotificationChannel.PAGERDUTY
            ]
        elif severity == AlertSeverity.WARNING:
            return [
                NotificationChannel.EMAIL,
                NotificationChannel.SLACK
            ]
        else:
            return [NotificationChannel.SLACK]
    
    def _get_email_recipients(self, alert: Alert) -> List[str]:
        """Get email recipients based on alert labels."""
        # Default recipients
        recipients = ["platform-team@company.com"]
        
        # Add team-specific recipients
        team = alert.labels.get("team")
        if team == "product":
            recipients.append("product-team@company.com")
        elif team == "security":
            recipients.append("security-team@company.com")
        
        # Add service-specific recipients
        service = alert.labels.get("service")
        if service == "database":
            recipients.append("dba-team@company.com")
        
        return recipients
    
    def _load_templates(self) -> Dict[NotificationChannel, NotificationTemplate]:
        """Load notification templates."""
        return {
            NotificationChannel.EMAIL: NotificationTemplate(
                subject_template="[{severity}] {alert_name} - {service}",
                body_template="""
Alert: {alert_name}
Severity: {severity}
Service: {service}
Environment: {environment}
Description: {description}
Timestamp: {timestamp}

Runbook: {runbook_url}
Dashboard: {dashboard_url}

This is an automated alert from the Agentic RAG monitoring system.
                """.strip(),
                channel=NotificationChannel.EMAIL
            ),
            NotificationChannel.SLACK: NotificationTemplate(
                subject_template="{severity} Alert: {alert_name}",
                body_template="""
*Service:* {service}
*Environment:* {environment}
*Description:* {description}
*Runbook:* {runbook_url}
                """.strip(),
                channel=NotificationChannel.SLACK
            ),
            NotificationChannel.WEBHOOK: NotificationTemplate(
                subject_template="{alert_name}",
                body_template="{description}",
                channel=NotificationChannel.WEBHOOK
            ),
            NotificationChannel.PAGERDUTY: NotificationTemplate(
                subject_template="{alert_name} - {service}",
                body_template="{description}",
                channel=NotificationChannel.PAGERDUTY
            )
        }
    
    def _load_escalation_rules(self) -> Dict[int, Dict[str, Any]]:
        """Load escalation rules."""
        return {
            1: {
                "delay": 15,  # minutes
                "channels": [NotificationChannel.EMAIL, NotificationChannel.SLACK]
            },
            2: {
                "delay": 30,  # minutes
                "channels": [NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY]
            },
            3: {
                "delay": 60,  # minutes
                "channels": [NotificationChannel.EMAIL, NotificationChannel.PAGERDUTY, NotificationChannel.SLACK]
            }
        }


# Global notification service instance
_notification_service: Optional[NotificationService] = None


async def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    global _notification_service
    
    if _notification_service is None:
        _notification_service = NotificationService()
        await _notification_service.start()
    
    return _notification_service

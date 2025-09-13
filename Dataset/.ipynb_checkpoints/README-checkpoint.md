# Atlassian Engagement & Retention — Synthetic Dataset

### users.csv
| column | description |
|---|---|
| user_id | Unique user identifier |
| signup_date | Date when user signed up |
| plan_tier | User's subscription plan |
| company_size | Size category of user's company |
| region | Geographic region |
| industry | Business industry category |
| acquisition_channel | How user was acquired |
| is_enterprise | Whether user is enterprise customer |
| churned_30d | User churned within 30 days |
| churned_90d | User churned within 90 days |
| downgraded | User downgraded their plan |
| expansion_event | User expanded their usage |

### sessions.csv
| column | notes |
|---|---|
| session_id | Unique session identifier |
| user_id | User identifier |
| session_start / session_end | Session start and end timestamps |
| device | Device type used |
| os | Operating system |
| app_version | Application version |
| country | Country code |
| session_length_sec | Session duration in seconds |

### events.csv
| column | notes |
|---|---|
| event_id | Unique event identifier |
| user_id | User identifier |
| session_id | Session identifier |
| ts | Event timestamp |
| feature_name | Feature being used |
| action | Type of action performed |
| duration_ms | Event duration in milliseconds |
| latency_ms | Response latency in milliseconds |
| success | Whether event completed successfully |

### billing.csv
| column | notes |
|---|---|
| user_id | User identifier |
| month | Billing month |
| plan_tier | Current subscription plan |
| active_seats | Number of active seats |
| mrr | Monthly recurring revenue |
| discount_applied | Whether discount was applied |
| invoices_overdue | Whether invoices are overdue |
| support_ticket_count | Number of support tickets |

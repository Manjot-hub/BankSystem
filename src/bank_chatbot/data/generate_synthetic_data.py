"""
Synthetic Banking Data Generator

Generates realistic but fake banking data for development and testing.
Includes: accounts, transactions, policies, FAQs, user profiles.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from faker import Faker
from faker.providers import BaseProvider


fake = Faker()


class BankingProvider(BaseProvider):
    """Custom Faker provider for banking data."""

    ACCOUNT_TYPES = ["checking", "savings", "money_market", "cd", "ira", "brokerage"]
    TRANSACTION_TYPES = ["debit", "credit", "transfer", "deposit", "withdrawal", "fee", "interest"]
    TRANSACTION_CATEGORIES = [
        "groceries", "dining", "transportation", "shopping", "entertainment",
        "healthcare", "utilities", "rent", "salary", "transfer_in", "transfer_out",
        "atm_withdrawal", "online_payment", "subscription", "insurance", "investment"
    ]
    MERCHANTS = [
        "Amazon", "Walmart", "Target", "Costco", "Starbucks", "Uber", "Lyft",
        "Shell", "Exxon", "Chevron", "Netflix", "Spotify", "Apple", "Google",
        "Microsoft", "Adobe", "Verizon", "AT&T", "T-Mobile", "Comcast",
        "Whole Foods", "Trader Joe's", "Kroger", "Safeway", "Publix"
    ]

    def account_type(self) -> str:
        return random.choice(self.ACCOUNT_TYPES)

    def transaction_type(self) -> str:
        return random.choice(self.TRANSACTION_TYPES)

    def transaction_category(self) -> str:
        return random.choice(self.TRANSACTION_CATEGORIES)

    def merchant(self) -> str:
        return random.choice(self.MERCHANTS)

    def routing_number(self) -> str:
        return f"{random.randint(100000000, 999999999)}"

    def account_number(self) -> str:
        return f"{random.randint(1000000000, 9999999999)}"

    def swift_code(self) -> str:
        banks = ["CHASUS33", "BOFAUS3N", "CITIUS33", "WELLSFARGO", "USBKUS44"]
        return random.choice(banks)

    def iban(self) -> str:
        return f"US{random.randint(10, 99)} {fake.bban()}"


fake.add_provider(BankingProvider)


class SyntheticDataGenerator:
    """Generates synthetic banking data for chatbot training and evaluation."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        fake.seed_instance(seed)
        self.output_dir = Path("data/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_users(self, count: int = 100) -> list[dict[str, Any]]:
        """Generate synthetic user profiles."""
        users = []
        for i in range(count):
            user = {
                "user_id": f"usr_{i+1:06d}",
                "email": fake.email(),
                "phone": fake.phone_number(),
                "full_name": fake.name(),
                "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
                "address": {
                    "street": fake.street_address(),
                    "city": fake.city(),
                    "state": fake.state_abbr(),
                    "zip_code": fake.zipcode(),
                    "country": "US"
                },
                "kyc_status": random.choice(["verified", "pending", "expired"]),
                "risk_rating": random.choice(["low", "medium", "high"]),
                "segment": random.choice(["retail", "premium", "private", "business"]),
                "created_at": fake.date_time_between(start_date="-5y", end_date="now").isoformat(),
                "last_login": fake.date_time_between(start_date="-30d", end_date="now").isoformat(),
            }
            users.append(user)
        return users

    def generate_accounts(self, users: list[dict], accounts_per_user: int = 2) -> list[dict[str, Any]]:
        """Generate synthetic accounts for users."""
        accounts = []
        for user in users:
            for _ in range(random.randint(1, accounts_per_user)):
                acc_type = fake.account_type()
                account = {
                    "account_id": f"acc_{fake.uuid4()[:12]}",
                    "user_id": user["user_id"],
                    "account_type": acc_type,
                    "account_name": f"{acc_type.title()} Account",
                    "currency": "USD",
                    "balance": round(random.uniform(100, 500000), 2),
                    "available_balance": round(random.uniform(100, 500000), 2),
                    "status": random.choice(["active", "active", "active", "frozen", "closed"]),
                    "routing_number": fake.routing_number(),
                    "account_number": fake.account_number(),
                    "interest_rate": round(random.uniform(0.01, 5.0), 4) if acc_type in ["savings", "money_market", "cd"] else 0.0,
                    "opened_date": fake.date_time_between(start_date="-10y", end_date="-30d").isoformat(),
                    "last_activity": fake.date_time_between(start_date="-7d", end_date="now").isoformat(),
                }
                accounts.append(account)
        return accounts

    def generate_transactions(self, accounts: list[dict], txns_per_account: int = 50) -> list[dict[str, Any]]:
        """Generate synthetic transactions for accounts."""
        transactions = []
        for account in accounts:
            if account["status"] != "active":
                continue
            for _ in range(random.randint(10, txns_per_account)):
                txn_type = fake.transaction_type()
                amount = round(random.uniform(1, 10000), 2)
                if txn_type in ["debit", "withdrawal", "fee", "transfer_out"]:
                    amount = -abs(amount)
                else:
                    amount = abs(amount)

                txn = {
                    "transaction_id": f"txn_{fake.uuid4()[:12]}",
                    "account_id": account["account_id"],
                    "user_id": None,  # Will be filled from account
                    "transaction_type": txn_type,
                    "category": fake.transaction_category(),
                    "merchant": fake.merchant() if txn_type in ["debit", "credit"] else None,
                    "amount": amount,
                    "currency": "USD",
                    "description": f"{txn_type.title()} - {fake.transaction_category()}",
                    "status": random.choice(["posted", "posted", "posted", "pending", "failed"]),
                    "timestamp": fake.date_time_between(start_date="-90d", end_date="now").isoformat(),
                    "balance_after": round(account["balance"] + amount, 2),
                }
                transactions.append(txn)
        return transactions

    def generate_policies(self) -> list[dict[str, Any]]:
        """Generate banking policy documents for RAG."""
        policies = [
            {
                "doc_id": "pol_001",
                "title": "Funds Availability Policy",
                "doc_type": "policy",
                "product": "all",
                "jurisdiction": "US",
                "version": "2024.1",
                "effective_date": "2024-01-01",
                "content": """
Funds Availability Policy

1. General Availability
Cash deposits, wire transfers, and the first $225 of a day's non-cash deposits are available on the first business day after deposit.

2. Case-by-Case Delays
We may delay availability up to 7 business days for:
- New accounts (open < 30 days)
- Large deposits (> $5,525)
- Redeposited checks
- Repeated overdraft accounts
- Emergency conditions

3. Check Hold Schedule
- Local checks: 2 business days
- Non-local checks: 5 business days
- Treasury checks: Next business day
- Cashier's checks: Next business day

4. Exceptions
Federal regulations (Reg CC) govern maximum hold periods. Contact us for specific cases.
""",
                "metadata": {"topic": "funds_availability", "keywords": ["hold", "deposit", "availability", "check"]}
            },
            {
                "doc_id": "pol_002",
                "title": "Wire Transfer Policy",
                "doc_type": "policy",
                "product": "all",
                "jurisdiction": "US",
                "version": "2024.1",
                "effective_date": "2024-01-01",
                "content": """
Wire Transfer Policy

1. Domestic Wires
- Cutoff: 4:00 PM ET for same-day processing
- Fee: $25 outgoing, $15 incoming
- Limits: $100,000 daily (retail), $1,000,000 (premium)

2. International Wires
- Cutoff: 3:00 PM ET
- Fee: $45 outgoing, $20 incoming
- SWIFT required for all international transfers
- Additional intermediary bank fees may apply

3. Information Required
- Recipient full name and address
- Recipient bank name, address, SWIFT/BIC
- Recipient account number/IBAN
- Purpose of payment
- Your account number

4. Cancellations
Once processed, wires cannot be cancelled. Contact recipient bank for recall (fees apply).
""",
                "metadata": {"topic": "wire_transfer", "keywords": ["wire", "transfer", "swift", "international", "domestic"]}
            },
            {
                "doc_id": "pol_003",
                "title": "Dispute Resolution Process",
                "doc_type": "policy",
                "product": "all",
                "jurisdiction": "US",
                "version": "2024.1",
                "effective_date": "2024-01-01",
                "content": """
Dispute Resolution Process

1. Unauthorized Transactions (Reg E)
- Report within 60 days of statement date
- Zero liability if reported within 2 business days
- Up to $50 liability if reported within 60 days
- Unlimited liability after 60 days

2. Billing Errors (Reg Z)
- Report within 60 days of statement
- We must acknowledge within 10 business days
- Resolution within 90 days (2 billing cycles)

3. Process
a. Contact us immediately via phone, app, or branch
b. We provisionally credit within 10 business days
c. Investigation completes within 45 days (90 for POS/foreign)
d. Final determination with explanation

4. Merchant Disputes
Attempt resolution with merchant first. Provide documentation.
""",
                "metadata": {"topic": "disputes", "keywords": ["dispute", "unauthorized", "fraud", "chargeback", "reg e", "reg z"]}
            },
            {
                "doc_id": "pol_004",
                "title": "Account Security & Authentication",
                "doc_type": "policy",
                "product": "all",
                "jurisdiction": "US",
                "version": "2024.2",
                "effective_date": "2024-06-01",
                "content": """
Account Security & Authentication Policy

1. Multi-Factor Authentication (MFA)
- Required for all digital banking access
- Options: SMS, Authenticator App, Security Key, Biometric
- At least one factor must be "something you have"

2. Session Management
- Idle timeout: 10 minutes (mobile), 5 minutes (web)
- Maximum session: 8 hours
- Re-authentication for sensitive actions

3. Device Registration
- Trusted devices remember for 90 days
- New device requires MFA + email confirmation
- Maximum 5 trusted devices per user

4. Fraud Monitoring
- Real-time transaction scoring
- Automatic block for high-risk patterns
- SMS/email alerts for suspicious activity

5. Account Recovery
- Identity verification required
- In-person at branch with government ID
- Or video verification with two documents
""",
                "metadata": {"topic": "security", "keywords": ["mfa", "authentication", "security", "fraud", "device", "recovery"]}
            },
            {
                "doc_id": "pol_005",
                "title": "Overdraft & Fee Schedule",
                "doc_type": "policy",
                "product": "checking",
                "jurisdiction": "US",
                "version": "2024.1",
                "effective_date": "2024-01-01",
                "content": """
Overdraft & Fee Schedule

1. Overdraft Fees
- Per item: $35 (max 4 per day = $140)
- Continuous overdraft (>5 days): $10/day
- Overdraft protection transfer: $12.50

2. Monthly Maintenance
- Basic Checking: $12 (waived with $1,500 avg balance or $500 direct deposit)
- Premium Checking: $25 (waived with $15,000 combined balances)
- Student Checking: $0 (age 17-24)
- Senior Checking: $0 (age 65+)

3. ATM Fees
- In-network: $0
- Out-of-network US: $3.00
- International: $5.00 + 3% currency conversion

4. Other Fees
- Stop payment: $30
- Cashier's check: $10
- Money order: $5
- Paper statement: $3
- Account research: $25/hour
""",
                "metadata": {"topic": "fees", "keywords": ["overdraft", "fee", "maintenance", "atm", "stop payment"]}
            }
        ]
        return policies

    def generate_faqs(self) -> list[dict[str, Any]]:
        """Generate FAQ documents for RAG."""
        faqs = [
            {
                "doc_id": "faq_001",
                "title": "How do I reset my online banking password?",
                "doc_type": "faq",
                "product": "digital",
                "content": """
To reset your password:
1. Go to the login page and click "Forgot Password"
2. Enter your username and registered email/phone
3. Choose verification method (SMS, email, or authenticator)
4. Enter the code and create a new password
5. Password must be 12+ characters with upper, lower, number, symbol

If you don't have access to your verification methods, visit a branch with government ID.
""",
                "metadata": {"topic": "password_reset", "keywords": ["password", "reset", "login", "forgot", "mfa"]}
            },
            {
                "doc_id": "faq_002",
                "title": "What are the mobile check deposit limits?",
                "doc_type": "faq",
                "product": "digital",
                "content": """
Mobile Check Deposit Limits:
- Daily limit: $5,000
- Monthly limit: $15,000
- Per check limit: $5,000
- Funds availability: Next business day for first $225
- Endorse: "For Mobile Deposit Only" + your signature
- Keep check for 14 days after deposit

Limits may vary by account type and history. Premium accounts have higher limits.
""",
                "metadata": {"topic": "mobile_deposit", "keywords": ["mobile", "check", "deposit", "limit", "endorse"]}
            },
            {
                "doc_id": "faq_003",
                "title": "How do I set up direct deposit?",
                "doc_type": "faq",
                "product": "checking",
                "content": """
To set up direct deposit:
1. Get your account and routing numbers from the app (Account Details)
2. Provide to your employer's payroll department
3. Or download pre-filled form from app > Account > Direct Deposit
4. First deposit typically takes 1-2 pay cycles

Benefits: Early access (up to 2 days), waives monthly fees on eligible accounts.
""",
                "metadata": {"topic": "direct_deposit", "keywords": ["direct", "deposit", "payroll", "routing", "account number"]}
            },
            {
                "doc_id": "faq_004",
                "title": "What is the difference between ACH and wire transfer?",
                "doc_type": "faq",
                "product": "all",
                "content": """
ACH vs Wire Transfer:

ACH (Automated Clearing House):
- Cost: Usually free
- Speed: 1-3 business days
- Limits: Lower daily/monthly limits
- Reversible: Yes (within 60 days for unauthorized)
- Use for: Payroll, bills, recurring transfers

Wire Transfer:
- Cost: $25 domestic, $45 international
- Speed: Same day (domestic), 1-2 days (international)
- Limits: Higher limits
- Reversible: Very difficult once sent
- Use for: Large purchases, real estate, urgent transfers

For most personal transfers, ACH is recommended.
""",
                "metadata": {"topic": "ach_vs_wire", "keywords": ["ach", "wire", "transfer", "difference", "cost", "speed"]}
            },
            {
                "doc_id": "faq_005",
                "title": "How do I report a lost or stolen debit card?",
                "doc_type": "faq",
                "product": "debit_card",
                "content": """
Report Lost/Stolen Debit Card Immediately:

1. In App: Cards > Select Card > Report Lost/Stolen
2. Phone: 1-800-XXX-XXXX (24/7)
3. Card is instantly deactivated
4. Replacement mailed in 5-7 business days
5. Expedited shipping: $15 (2-3 days)
6. Digital card available immediately in mobile wallet

Zero liability for unauthorized transactions if reported promptly.
Monitor account for suspicious activity.
""",
                "metadata": {"topic": "lost_card", "keywords": ["lost", "stolen", "debit", "card", "report", "replacement"]}
            }
        ]
        return faqs

    def generate_all(self) -> dict[str, Any]:
        """Generate all synthetic data."""
        print("Generating users...")
        users = self.generate_users(100)

        print("Generating accounts...")
        accounts = self.generate_accounts(users, 3)

        print("Generating transactions...")
        transactions = self.generate_transactions(accounts, 100)

        print("Generating policies...")
        policies = self.generate_policies()

        print("Generating FAQs...")
        faqs = self.generate_faqs()

        # Attach user_id to transactions
        account_to_user = {a["account_id"]: a["user_id"] for a in accounts}
        for txn in transactions:
            txn["user_id"] = account_to_user.get(txn["account_id"])

        data = {
            "users": users,
            "accounts": accounts,
            "transactions": transactions,
            "policies": policies,
            "faqs": faqs,
            "generated_at": datetime.now().isoformat(),
            "stats": {
                "users": len(users),
                "accounts": len(accounts),
                "transactions": len(transactions),
                "policies": len(policies),
                "faqs": len(faqs),
            }
        }

        # Save to files
        for key, value in data.items():
            if key != "stats" and key != "generated_at":
                filepath = self.output_dir / f"{key}.json"
                with open(filepath, "w") as f:
                    json.dump(value, f, indent=2)
                print(f"Saved {len(value)} {key} to {filepath}")

        # Save combined
        with open(self.output_dir / "all_data.json", "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nGeneration complete. Stats: {data['stats']}")
        return data


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    generator.generate_all()
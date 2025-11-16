"""
Human-in-the-Middle (HITM) System - Object-Oriented Implementation
Enhanced version with robust API key management

This module provides a complete HITM system for AI response quality control.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from pathlib import Path


# ============================================================================
# DISABLE LANGSMITH TRACING
# ============================================================================
# LangChain packages automatically send telemetry to LangSmith if env vars exist
# This prevents 'API key has expired' errors from LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def locating_environment_file() -> Optional[Path]:
    """
    Search upward for keys.env or .env file.

    Returns:
        Path to environment file or None if not found
    """
    file_names = ("keys.env", ".env")
    current = Path.cwd()

    # Search up to 5 levels up
    for _ in range(5):
        for name in file_names:
            candidate = current / name
            if candidate.exists():
                print(f"  Found environment file: {candidate}")
                return candidate
        current = current.parent

    return None


def load_api_keys(env_path: Optional[Path] = None, required: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """
    Load environment variables and return them as a dictionary.
    Automatically populates os.environ.

    Args:
        env_path: Path to environment file (auto-detects if None)
        required: List of required key names (loads all if None)

    Returns:
        Dictionary of key names to values
    """
    # Auto-detect environment file if not provided
    if env_path is None:
        env_path = locating_environment_file()

    if env_path is None:
        print("❌ No keys.env or .env found in current or parent directories.")
        return {}

    # Load into os.environ
    load_dotenv(env_path, override=True)
    print(f"  Loaded environment from: {env_path}")

    # If no required list provided, read all keys from env file
    if required is None:
        required = [
            line.split("=")[0].strip()
            for line in env_path.read_text().splitlines()
            if "=" in line and not line.strip().startswith("#") and line.strip()
        ]

    keys_status = {}

    print("\n🔍 Checking API keys:")
    for key in required:
        value = os.getenv(key)
        if value:
            # Mask the key for security
            masked_value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"    {key}: {masked_value}") 
            keys_status[key] = value
        else:
            print(f"     {key}: MISSING")
            keys_status[key] = None

    return keys_status


def validate_anthropic_api_key() -> bool:
    """
    Validate that ANTHROPIC_API_KEY is available and properly formatted.

    Returns:
        True if valid, False otherwise
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n❌ ANTHROPIC_API_KEY not found!")
        print("\nPlease set it using one of these methods:")
        print("1. Create a .env or keys.env file with:")
        print("   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("\n2. Set environment variable:")
        print("   export ANTHROPIC_API_KEY=your-key  # Linux/Mac")
        print("   set ANTHROPIC_API_KEY=your-key     # Windows")
        print("\n3. Get your API key from: https://console.anthropic.com/")
        return False

    # Validate format (Anthropic keys typically start with 'sk-ant-')
    if not api_key.startswith('sk-ant-'):
        print(f"\n   Warning: API key format unexpected")
        print(f"   Expected to start with 'sk-ant-'")
        print(f"   Found: {api_key[:10]}...")
        # Don't fail - might be a valid key with different format

    print(f"\n  ANTHROPIC_API_KEY validated: {api_key[:15]}...")
    return True

def validate_openai_api_key() -> bool:
    """
    Validate that OPENAI_API_KEY is available and properly formatted.

    Returns:
        True if valid, False otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("\n  OPENAI_API_KEY not found!")
        print("\n Please set it using one of these methods:")
        print("1. Create a .env or keys.env file with:")
        print("   OPENAI_API_KEY=sk-xxxxxx")
        print("\n 2. Set environment variable:")
        print("   export OPENAI_API_KEY=your-key   # Linux/Mac")
        print("   set OPENAI_API_KEY=your-key      # Windows")
        print("\n 3. Get your API key from: https://platform.openai.com/api-keys")
        return False

    # Common valid OpenAI key prefixes
    valid_prefixes = (
        "sk-",
        "sk-test-",
        "sk-live-",
        "sk-proj-",
        "sk-or-",
    )

    if not api_key.startswith(valid_prefixes):
        print("\n   Warning: API key format unexpected")
        print(f"   Expected to start with one of: {valid_prefixes}")
        print(f"   Found: {api_key[:10]}...")
        # Do NOT fail — format may change

    print(f"\n  OPENAI_API_KEY validated: {api_key[:15]}...")
    return True

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Request:
    """Represents a user request with metadata."""

    id: str
    user_query: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AIResponse:
    """Represents an AI-generated response."""

    request_id: str
    content: str
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class QualityCheck:
    """Results from automated quality checks."""

    passed: bool
    flags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class HumanReview:
    """Human review decision and feedback."""

    request_id: str
    decision: Literal["approve", "edit", "reject"]
    edited_content: str = ""
    feedback: str = ""
    reviewer: str = "human_reviewer"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


# ============================================================================
# CONFIGURATION
# ============================================================================

class HITMConfig:
    """Configuration for HITM system."""

    def __init__(
            self,
            # always_review: bool = False,
            model_name: str = "claude-sonnet-4-20250514",
            temperature: float = 0.7,
            max_tokens: int = 1024,
            judge_temperature: float = 0.0,
            judge_max_tokens: int = 512,
            log_file: str = "hitm_feedback.jsonl",
            auto_approve_on_pass: bool = True,
            api_key: Optional[str] = None
    ):
        """
        Initialize configuration.

        Args:
            model_name: Claude model to use
            temperature: Response randomness (0-1)
            max_tokens: Maximum response length
            judge_temperature: Judge model temperature
            judge_max_tokens: Judge max tokens
            log_file: Path to log file
            auto_approve_on_pass: Auto-approve passed checks
            api_key: Optional API key to set (overrides env var)
        """
        # Handle API key if provided
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            print("  API key set from config parameter")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        self.log_file = log_file
        self.auto_approve_on_pass = auto_approve_on_pass

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'HITMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'judge_temperature': self.judge_temperature,
            'judge_max_tokens': self.judge_max_tokens,
            'log_file': self.log_file,
            'auto_approve_on_pass': self.auto_approve_on_pass
        }


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class AIResponseGenerator:
    """Generates AI responses to user queries using LangChain."""

    def __init__(self, config: HITMConfig):
        """
        Initialize the response generator.

        Args:
            config: HITM system configuration

        Raises:
            ValueError: If API key is not available
        """
        self.config = config

        # Validate API key before initializing LLM
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please load it before initializing AIResponseGenerator."
            )

        try:
            self.llm = ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            print(f"     Initialized LLM: {config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatAnthropic: {e}") from e

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."),
            ("human", "{query}")
        ])
        self.chain = self.prompt_template | self.llm

    def generate(self, request: Request) -> AIResponse:
        """
        Generate a response for the given request.

        Args:
            request: User request object

        Returns:
            AIResponse object containing the generated content
        """
        response = self.chain.invoke({"query": request.user_query})

        return AIResponse(
            request_id=request.id,
            content=response.content,
            model=self.config.model_name
        )

    def update_system_prompt(self, new_system_prompt: str) -> None:
        """Update the system prompt for response generation."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt),
            ("human", "{query}")
        ])
        self.chain = self.prompt_template | self.llm


class QualityChecker:
    """Performs automated quality checks using LLM as judge."""

    def __init__(self, config: HITMConfig):
        """
        Initialize the quality checker.

        Args:
            config: HITM system configuration

        Raises:
            ValueError: If API key is not available
        """
        self.config = config

        # Validate API key before initializing LLM
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please load it before initializing QualityChecker."
            )

        try:
            self.judge_llm = ChatAnthropic(
                model=config.model_name,
                temperature=config.judge_temperature,
                max_tokens=config.judge_max_tokens
            )
            print(f"     Initialized Judge LLM: {config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Judge ChatAnthropic: {e}") from e

        self.min_length = 10
        self.max_length = 2000

    def check(self, request: Request, response: AIResponse) -> QualityCheck:
        """
        Run automated quality checks on the response.

        Args:
            request: Original user request
            response: AI-generated response

        Returns:
            QualityCheck object with results
        """
        flags = []

        # Basic length checks
        if len(response.content) < self.min_length:
            flags.append("Response too short")

        if len(response.content) > self.max_length:
            flags.append("Response too long")

        # LLM-based quality evaluation
        reasoning = self._evaluate_with_llm(request, response, flags)

        # Determine if passed
        passed = len(flags) == 0

        return QualityCheck(
            passed=passed,
            flags=flags,
            reasoning=reasoning
        )

    def _evaluate_with_llm(
            self,
            request: Request,
            response: AIResponse,
            flags: List[str]
    ) -> str:
        """Evaluate response quality using LLM judge."""

        judge_prompt = f"""Evaluate this AI response for quality and safety.

User Query: {request.user_query}

AI Response: {response.content}

Check for:
1. Safety concerns (harmful content, misinformation)
2. Relevance to the query
3. Completeness of the answer
4. Professionalism and appropriateness

Respond in JSON format:
{{
    "safe": true/false,
    "relevant": true/false,
    "complete": true/false,
    "professional": true/false,
    "issues": ["list any specific issues"],
    "reasoning": "brief explanation"
}}"""

        try:
            judge_response = self.judge_llm.invoke([HumanMessage(content=judge_prompt)])

            # Parse JSON response
            content = judge_response.content
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())

            # Add issues to flags
            if not result.get("safe", True):
                flags.append("Safety concern")
            if not result.get("relevant", True):
                flags.append("Not relevant to query")
            if not result.get("complete", True):
                flags.append("Incomplete answer")
            if not result.get("professional", True):
                flags.append("Unprofessional tone")

            flags.extend(result.get("issues", []))
            reasoning = result.get("reasoning", "")

        except Exception as e:
            print(f"      Judge evaluation error: {e}")
            reasoning = "Could not complete automated review"

        return reasoning

    def set_length_limits(self, min_length: int, max_length: int) -> None:
        """Update length check limits."""
        self.min_length = min_length
        self.max_length = max_length


class ReviewInterface(ABC):
    """Abstract base class for review interfaces."""

    @abstractmethod
    def review(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> HumanReview:
        """Present item for review and get human decision."""
        pass


class ConsoleReviewInterface(ReviewInterface):
    """Console-based human review interface."""

    def review(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> HumanReview:
        """
        Present item for human review via console.

        Args:
            request: Original user request
            response: AI-generated response
            quality_check: Automated quality check results

        Returns:
            HumanReview object with decision
        """
        self._display_review_info(request, response, quality_check)

        decision = self._get_decision()
        edited_content = ""
        feedback = ""

        if decision == "edit":
            edited_content = self._get_edited_content(response.content)
            feedback = input("\n  Feedback (optional): ").strip()
        elif decision == "reject":
            feedback = input("\n  Reason for rejection: ").strip()
        elif decision == "approve":
            feedback = input("\n  Feedback (optional): ").strip()

        return HumanReview(
            request_id=request.id,
            decision=decision,
            edited_content=edited_content,
            feedback=feedback
        )

    def _display_review_info(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> None:
        """Display review information."""
        print("\n" + "=" * 80)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 80)

        print(f"\n  Request ID: {request.id}")
        print(f"  Time: {request.timestamp}")

        print(f"\n  USER QUERY:")
        print(f"   {request.user_query}")

        print(f"\n  AI RESPONSE:")
        print(f"   {response.content}")

        print(f"\n   QUALITY CHECK:")
        print(f"   Status: {'✓ PASS' if quality_check.passed else '✗ FLAGGED'}")
        if quality_check.flags:
            print(f"   Flags: {', '.join(quality_check.flags)}")
        if quality_check.reasoning:
            print(f"   Reasoning: {quality_check.reasoning}")

        print("\n" + "=" * 80)

    def _get_decision(self) -> str:
        """Get human decision."""
        while True:
            print("\nDECISION OPTIONS:")
            print("  [1] ✓ Approve - Send as is")
            print("  [2] ✏️  Edit - Modify the response")
            print("  [3] ✗ Reject - Regenerate response")

            choice = input("\nYour decision (1/2/3): ").strip()

            if choice == "1":
                return "approve"
            elif choice == "2":
                return "edit"
            elif choice == "3":
                return "reject"
            else:
                print("  Invalid choice. Please enter 1, 2, or 3.")

    def _get_edited_content(self, original: str) -> str:
        """Get edited content from human."""
        print("\n   EDIT MODE")
        print("Enter your edited version (press Enter twice when done):\n")

        lines = []
        empty_count = 0

        while empty_count < 2:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)

        edited = "\n".join(lines).strip()
        return edited if edited else original


class FeedbackLogger:
    """Logs all decisions and feedback for analysis."""

    def __init__(self, log_file: str = "hitm_feedback.jsonl"):
        """
        Initialize the feedback logger.

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file

    def log(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck,
            human_review: HumanReview
    ) -> None:
        """
        Log complete review cycle.

        Args:
            request: User request
            response: AI response
            quality_check: Quality check results
            human_review: Human review decision
        """
        entry = {
            "request_id": request.id,
            "timestamp": request.timestamp.isoformat(),
            "user_query": request.user_query,
            "ai_response": response.content,
            "model": response.model,
            "quality_check": quality_check.to_dict(),
            "human_review": human_review.to_dict()
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        print(f"\n  Logged to {self.log_file}")

    def get_stats(self) -> Dict:
        """
        Get statistics from logs.

        Returns:
            Dictionary containing statistics
        """
        if not os.path.exists(self.log_file):
            return {"total": 0}

        stats = {
            "total": 0,
            "approved": 0,
            "edited": 0,
            "rejected": 0,
            "flagged": 0
        }

        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                stats["total"] += 1
                stats[entry["human_review"]["decision"]] += 1
                if not entry["quality_check"]["passed"]:
                    stats["flagged"] += 1

        return stats

    def get_recent_entries(self, n: int = 10) -> List[Dict]:
        """
        Get recent log entries.

        Args:
            n: Number of recent entries to retrieve

        Returns:
            List of log entries
        """
        if not os.path.exists(self.log_file):
            return []

        with open(self.log_file, 'r') as f:
            lines = f.readlines()

        entries = [json.loads(line) for line in lines[-n:]]
        return entries


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class HITMSystem:
    """Complete Human-in-the-Middle system orchestrator."""

    def __init__(
            self,
            config: Optional[HITMConfig] = None,
            review_interface: Optional[ReviewInterface] = None,
            validate_api: bool = True
    ):
        """
        Initialize the HITM system.

        Args:
            config: System configuration (uses defaults if None)
            review_interface: Review interface (uses console if None)
            validate_api: Whether to validate API key on initialization

        Raises:
            ValueError: If API key validation fails
            RuntimeError: If component initialization fails
        """
        print("\n  Initializing HITM System...")

        # Validate API key if requested
        if validate_api:
            if not validate_anthropic_api_key():
                raise ValueError("API key validation failed")

        self.config = config or HITMConfig()

        # Initialize components with error handling
        try:
            print("\n  Initializing components...")
            self.generator = AIResponseGenerator(self.config)
            self.checker = QualityChecker(self.config)
            self.review_interface = review_interface or ConsoleReviewInterface()
            self.logger = FeedbackLogger(self.config.log_file)
            print("     Review interface ready")
            print("     Feedback logger ready")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize HITM system: {e}\n"
                f"This might be due to a missing or invalid API key."
            ) from e

        # Request tracking
        self.request_counter = 0

        print("\n  HITM System initialization complete!")

    def process(
            self,
            user_query: str,
            require_review: Optional[bool] = None,
            request_metadata: Optional[Dict] = None
    ) -> str:
        """
        Process a user query through the HITM pipeline.

        Args:
            user_query: The user's question/request
            require_review: Force human review (None=auto, True=always, False=never)
            request_metadata: Optional metadata for the request

        Returns:
            Final response to send to user
        """
        self.request_counter += 1

        # Step 1: Create request
        request = Request(
            id=f"req_{self.request_counter:04d}",
            user_query=user_query,
            metadata=request_metadata or {}
        )

        print(f"\n  Processing request {request.id}...")

        # Step 2: Generate AI response
        print("   → Generating AI response...")
        response = self.generator.generate(request)
        print("   ✓ Response generated")

        # Step 3: Automated quality check
        print("   → Running quality checks...")
        quality_check = self.checker.check(request, response)
        print(f"   {'✓ Passed' if quality_check.passed else '⚠ Flagged'}")

        # Step 4: Determine if human review needed
        needs_review = self._determine_review_needed(quality_check, require_review)

        if needs_review:
            final_response = self._handle_human_review(
                request, response, quality_check
            )
        else:
            final_response = self._handle_auto_approve(
                request, response, quality_check
            )

        # Display final response
        self._display_final_response(final_response)

        return final_response

    def _determine_review_needed(
            self,
            quality_check: QualityCheck,
            require_review: Optional[bool]
    ) -> bool:
        """Determine if human review is needed."""
        if require_review is not None:
            return require_review

        return not quality_check.passed

    def _handle_human_review(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> str:
        """Handle human review workflow."""
        print("   → Routing to human reviewer...")
        human_review = self.review_interface.review(request, response, quality_check)

        # Apply decision
        if human_review.decision == "approve":
            final_response = response.content
            print("\n  Response APPROVED")
        elif human_review.decision == "edit":
            final_response = human_review.edited_content
            print("\n   Response EDITED")
        else:  # reject
            final_response = "[Response rejected - regeneration needed]"
            print("\n  Response REJECTED")

        # Log feedback
        self.logger.log(request, response, quality_check, human_review)

        return final_response

    def _handle_auto_approve(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> str:
        """Handle auto-approval workflow."""
        final_response = response.content
        print("   ✓ Auto-approved (no review needed)")

        # Still log for tracking
        auto_review = HumanReview(
            request_id=request.id,
            decision="approve",
            feedback="Auto-approved",
            reviewer="system"
        )
        self.logger.log(request, response, quality_check, auto_review)

        return final_response

    def _display_final_response(self, final_response: str) -> None:
        """Display the final response."""
        print("\n" + "=" * 80)
        print("FINAL RESPONSE TO USER:")
        print("=" * 80)
        print(final_response)
        print("=" * 80 + "\n")

    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return self.logger.get_stats()

    def update_config(self, config: HITMConfig) -> None:
        """Update system configuration and reinitialize components."""
        self.config = config
        self.generator = AIResponseGenerator(config)
        self.checker = QualityChecker(config)
        self.logger = FeedbackLogger(config.log_file)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def display_statistics(stats: Dict) -> None:
    """Display system statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("HITM SYSTEM STATISTICS")
    print("=" * 60)
    print(f"Total Requests:     {stats.get('total', 0)}")
    print(f"Approved:           {stats.get('approved', 0)}")
    print(f"Edited:             {stats.get('edited', 0)}")
    print(f"Rejected:           {stats.get('rejected', 0)}")
    print(f"Flagged by System:  {stats.get('flagged', 0)}")

    if stats.get('total', 0) > 0:
        approval_rate = (stats.get('approved', 0) / stats['total']) * 100
        print(f"\nApproval Rate:      {approval_rate:.1f}%")
    print("=" * 60)


def run_interactive_demo(hitm_system: HITMSystem) -> None:
    """
    Run interactive demo of HITM system.

    Args:
        hitm_system: Initialized HITM system
    """
    print("\n" + "=" * 80)
    print("HUMAN IN THE MIDDLE - INTERACTIVE DEMO")
    print("=" * 80)
    print("\nEnter user queries to see the HITM system in action.")
    print("Type 'stats' to see statistics, or 'quit' to exit.\n")

    while True:
        user_input = input("\n  User Query (or 'quit'/'stats'): ").strip()

        if user_input.lower() == 'quit':
            print("\n  Goodbye!")
            break

        if user_input.lower() == 'stats':
            stats = hitm_system.get_statistics()
            display_statistics(stats)
            continue

        if not user_input:
            print("  Please enter a query.")
            continue

        try:
            # Force review in your query (Recommended for testing)
            hitm_system.process(user_input, require_review=True)

            # LLM as a Judge Evaluates the answer and sets review requirements
            # hitm_system.process(user_input)

        except KeyboardInterrupt:
            print("\n\n   Process interrupted. Returning to menu...")
            continue
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive API key management."""
    print("\n" + "=" * 80)
    print("HUMAN IN THE LOOP PIPELINE - INITIALIZATION")
    print("=" * 80)

    # Step 1: Locate and load environment file
    print("\n  Step 1: Locating environment file...")
    env_path = locating_environment_file()

    if env_path is None:
        print("\n No environment file found!")
        print("\n Please create a .env or keys.env file with:")
        print("ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("\n Alternatively, set the environment variable manually:")
        print("export ANTHROPIC_API_KEY=your-key  # Linux/Mac")
        print("set ANTHROPIC_API_KEY=your-key     # Windows")
        sys.exit(1)

    # Step 2: Load API keys
    print("\n  Step 2: Loading API keys...")
    keys = load_api_keys(env_path)

    # Check for required keys
    anthropic_key = keys.get("ANTHROPIC_API_KEY")
    openai_key = keys.get("OPENAI_API_KEY")

    if not anthropic_key:
        print("\n  ANTHROPIC_API_KEY is required but not found!")
        print("Please add it to your environment file.")
        sys.exit(1)

    # Optional: warn about OpenAI key if present but not used
    if openai_key:
        print("\n   Note: OPENAI_API_KEY found but not used by this system")

    # Step 3: Validate Anthropic API key
    print("\n  Step 3: Validating API key...")
    if not validate_anthropic_api_key():
        sys.exit(1)

    if not validate_openai_api_key():
        sys.exit(1)

    # Step 4: Create configuration
    print("\n   Step 4: Creating configuration...")
    config = HITMConfig(
        model_name="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1024,
        log_file="hitm_feedback.jsonl"
    )
    print("     Configuration created")

    # Step 5: Initialize HITM system
    print("\n  Step 5: Initializing HITM system...")
    try:
        hitm_system = HITMSystem(config=config, validate_api=True)
    except Exception as e:
        print(f"\n  Failed to initialize HITM system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  SYSTEM READY!")
    print("=" * 80)

    # Step 6: Run interactive demo
    print("\nStarting interactive demo...")
    run_interactive_demo(hitm_system)


if __name__ == "__main__":
    main()


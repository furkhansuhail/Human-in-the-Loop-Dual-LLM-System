"""
Human-in-the-Middle (HITM) System - Dual LLM Implementation
Enhanced version with Anthropic + OpenAI support and flexible role assignment

This module provides a complete HITM system where users can choose:
- Which LLM generates the response (Anthropic or OpenAI)
- The other LLM automatically becomes the judge
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
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
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
                print(f"Found environment file: {candidate}")
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
        print(" No keys.env or .env found in current or parent directories.")
        return {}

    # Load into os.environ
    load_dotenv(env_path, override=True)
    print(f" Loaded environment from: {env_path}")

    # If no required list provided, read all keys from env file
    if required is None:
        required = [
            line.split("=")[0].strip()
            for line in env_path.read_text().splitlines()
            if "=" in line and not line.strip().startswith("#") and line.strip()
        ]

    keys_status = {}

    print("\n Checking API keys:")
    for key in required:
        value = os.getenv(key)
        if value:
            # Mask the key for security
            masked_value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"  {key}: {masked_value}")
            keys_status[key] = value
        else:
            print(f"  {key}: MISSING")
            keys_status[key] = None

    return keys_status


def validate_anthropic_api_key() -> bool:
    """Validate that ANTHROPIC_API_KEY is available and properly formatted."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("\n ANTHROPIC_API_KEY not found!")
        print("\n Please set it using one of these methods:")
        print("1. Create a .env or keys.env file with:")
        print("   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("\n 2. Set environment variable:")
        print("   export ANTHROPIC_API_KEY=your-key  # Linux/Mac")
        print("   set ANTHROPIC_API_KEY=your-key     # Windows")
        print("\n 3. Get your API key from: https://console.anthropic.com/")
        return False

    if not api_key.startswith('sk-ant-'):
        print(f"\n Warning: API key format unexpected")
        print(f"   Expected to start with 'sk-ant-'")
        print(f"   Found: {api_key[:10]}...")

    print(f"\n ANTHROPIC_API_KEY validated: {api_key[:15]}...")
    return True


def validate_openai_api_key() -> bool:
    """Validate that OPENAI_API_KEY is available and properly formatted."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("\n OPENAI_API_KEY not found!")
        print("\n Please set it using one of these methods:")
        print("1. Create a .env or keys.env file with:")
        print("   OPENAI_API_KEY=sk-xxxxxx")
        print("\n 2. Set environment variable:")
        print("   export OPENAI_API_KEY=your-key   # Linux/Mac")
        print("   set OPENAI_API_KEY=your-key      # Windows")
        print("\n 3. Get your API key from: https://platform.openai.com/api-keys")
        return False

    valid_prefixes = ("sk-", "sk-test-", "sk-live-", "sk-proj-", "sk-or-")

    if not api_key.startswith(valid_prefixes):
        print("\n Warning: API key format unexpected")
        print(f"  Expected to start with one of: {valid_prefixes}")
        print(f"  Found: {api_key[:10]}...")

    print(f"\n OPENAI_API_KEY validated: {api_key[:15]}...")
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
    provider: str  # 'anthropic' or 'openai'
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
    judge_provider: str  # Which LLM did the judging
    judge_model: str
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


@dataclass
class HITMConfig:
    """Configuration for HITM system with dual LLM support."""

    # Generator settings
    generator_provider: Literal["anthropic", "openai"] = "anthropic"
    generator_model: str = "claude-sonnet-4-20250514"
    generator_temperature: float = 0.7
    generator_max_tokens: int = 1024

    # Judge settings (automatically set to opposite of generator)
    judge_provider: Literal["anthropic", "openai"] = "openai"
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.3
    judge_max_tokens: int = 1024

    # System settings
    log_file: str = "hitm_feedback.jsonl"
    enable_quality_check: bool = True


# ============================================================================
# LLM FACTORY
# ============================================================================

class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(provider: str, model: str, temperature: float, max_tokens: int):
        """
        Create an LLM instance based on provider.

        Args:
            provider: 'anthropic' or 'openai'
            model: Model name
            temperature: Temperature setting
            max_tokens: Max tokens to generate

        Returns:
            LangChain LLM instance
        """
        if provider == "anthropic":
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'anthropic' or 'openai'")


# ============================================================================
# CORE COMPONENTS
# ============================================================================

class AIResponseGenerator:
    """Generates AI responses using configurable LLM."""

    def __init__(self, config: HITMConfig):
        """Initialize with configuration."""
        self.config = config
        self.llm = LLMFactory.create_llm(
            provider=config.generator_provider,
            model=config.generator_model,
            temperature=config.generator_temperature,
            max_tokens=config.generator_max_tokens
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."),
            ("human", "{query}")
        ])

    def generate(self, request: Request) -> AIResponse:
        """Generate AI response for a request."""
        chain = self.prompt | self.llm
        result = chain.invoke({"query": request.user_query})

        response = AIResponse(
            request_id=request.id,
            content=result.content,
            model=self.config.generator_model,
            provider=self.config.generator_provider,
            metadata={
                "temperature": self.config.generator_temperature,
                "max_tokens": self.config.generator_max_tokens
            }
        )

        return response


class QualityChecker:
    """Checks response quality using a different LLM as judge."""

    def __init__(self, config: HITMConfig):
        """Initialize with configuration."""
        self.config = config
        self.judge_llm = LLMFactory.create_llm(
            provider=config.judge_provider,
            model=config.judge_model,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens
        )

        self.eval_prompt = (ChatPromptTemplate.from_messages
        ([
            ("system", """You are an AI response quality evaluator. Analyze the following AI response and determine if it meets quality standards.
            Evaluation criteria:
            1. Accuracy: Is the information correct?
            2. Helpfulness: Does it answer the question?
            3. Safety: Is it free from harmful content?
            4. Clarity: Is it well-written and clear?
            
            Respond in JSON format:
            {{
            "passed": true/false,
                "confidence": 0.0-1.0,
                    "flags": ["flag1", "flag2"],
                        "reasoning": "brief explanation"
                        }}"""),
            ("human", """Original Query: {query}
            
            AI Response: {response}
            Evaluate this response:""")
        ]))

    def check(self, request: Request, response: AIResponse) -> QualityCheck:
        """Perform quality check on AI response."""
        if not self.config.enable_quality_check:
            return QualityCheck(
                passed=True,
                judge_provider=self.config.judge_provider,
                judge_model=self.config.judge_model,
                reasoning="Quality check disabled"
            )

        # Get evaluation from judge LLM
        chain = self.eval_prompt | self.judge_llm
        result = chain.invoke({
            "query": request.user_query,
            "response": response.content
        })

        # Parse JSON response
        try:
            eval_result = json.loads(result.content)
            return QualityCheck(
                passed=eval_result.get("passed", True),
                judge_provider=self.config.judge_provider,
                judge_model=self.config.judge_model,
                confidence=eval_result.get("confidence", 1.0),
                flags=eval_result.get("flags", []),
                reasoning=eval_result.get("reasoning", ""),
                metadata={"raw_evaluation": result.content}
            )
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return QualityCheck(
                passed=True,
                judge_provider=self.config.judge_provider,
                judge_model=self.config.judge_model,
                confidence=0.5,
                reasoning="Could not parse evaluation JSON",
                metadata={"raw_response": result.content}
            )


class HumanReviewInterface:
    """Interface for human review of AI responses."""

    def review(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck
    ) -> HumanReview:
        """Present response to human for review."""
        print("\n" + "=" * 80)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 80)

        print(f"\n Original Query:")
        print(f"   {request.user_query}")

        print(f"\n AI Response (from {response.provider} - {response.model}):")
        print(f"   {response.content}")

        print(f"\n Quality Check (by {quality_check.judge_provider} - {quality_check.judge_model}):")
        print(f"   Status: {'✅ PASSED' if quality_check.passed else '❌ FAILED'}")
        print(f"   Confidence: {quality_check.confidence:.2f}")
        if quality_check.flags:
            print(f"   Flags: {', '.join(quality_check.flags)}")
        print(f"   Reasoning: {quality_check.reasoning}")

        print("\n" + "-" * 80)
        print("Review Options:")
        print("  [1] Approve - Send response as-is")
        print("  [2] Edit    - Modify the response")
        print("  [3] Reject  - Reject and mark for regeneration")
        print("-" * 80)

        while True:
            choice = input("\nYour decision [1/2/3]: ").strip()

            if choice == "1":
                feedback = input("Optional feedback: ").strip()
                return HumanReview(
                    request_id=request.id,
                    decision="approve",
                    feedback=feedback or "Approved by human reviewer"
                )

            elif choice == "2":
                print("\nEnter edited response (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "" and len(lines) > 0 and lines[-1] == "":
                        break
                    lines.append(line)

                edited_content = "\n".join(lines[:-1])  # Remove last empty line
                feedback = input("\nFeedback on edits: ").strip()

                return HumanReview(
                    request_id=request.id,
                    decision="edit",
                    edited_content=edited_content,
                    feedback=feedback or "Edited by human reviewer"
                )

            elif choice == "3":
                feedback = input("Reason for rejection: ").strip()
                return HumanReview(
                    request_id=request.id,
                    decision="reject",
                    feedback=feedback or "Rejected by human reviewer"
                )

            else:
                print(" Invalid choice. Please enter 1, 2, or 3.")


class FeedbackLogger:
    """Logs all interactions and feedback to a file."""

    def __init__(self, log_file: str):
        """Initialize logger with file path."""
        self.log_file = Path(log_file)
        self.stats = {
            "total": 0,
            "approved": 0,
            "edited": 0,
            "rejected": 0,
            "flagged": 0
        }

    def log(
            self,
            request: Request,
            response: AIResponse,
            quality_check: QualityCheck,
            human_review: HumanReview
    ) -> None:
        """Log a complete interaction."""
        entry = {
            "request": request.to_dict(),
            "response": response.to_dict(),
            "quality_check": quality_check.to_dict(),
            "human_review": human_review.to_dict()
        }

        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # Update statistics
        self.stats["total"] += 1
        if human_review.decision == "approve":
            self.stats["approved"] += 1
        elif human_review.decision == "edit":
            self.stats["edited"] += 1
        elif human_review.decision == "reject":
            self.stats["rejected"] += 1

        if not quality_check.passed:
            self.stats["flagged"] += 1

    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.copy()


# ============================================================================
# MAIN HITM SYSTEM
# ============================================================================

class HITMSystem:
    """Main Human-in-the-Middle system orchestrator."""

    def __init__(self, config: HITMConfig, validate_api: bool = True):
        """
        Initialize HITM system.

        Args:
            config: System configuration
            validate_api: Whether to validate API keys on init
        """
        self.config = config

        if validate_api:
            self._validate_apis()

        # Initialize components
        self.generator = AIResponseGenerator(config)
        self.checker = QualityChecker(config)
        self.review_interface = HumanReviewInterface()
        self.logger = FeedbackLogger(config.log_file)

        print("\n  HITM System initialized")
        print(f"   Generator: {config.generator_provider} ({config.generator_model})")
        print(f"   Judge: {config.judge_provider} ({config.judge_model})")

    def _validate_apis(self) -> None:
        """Validate required API keys are present."""
        if self.config.generator_provider == "anthropic" or self.config.judge_provider == "anthropic":
            if not validate_anthropic_api_key():
                raise ValueError("Anthropic API key validation failed")

        if self.config.generator_provider == "openai" or self.config.judge_provider == "openai":
            if not validate_openai_api_key():
                raise ValueError("OpenAI API key validation failed")

    def process(self, user_query: str, require_review: Optional[bool] = None) -> str:
        """
        Process a user query through the HITM pipeline.

        Args:
            user_query: The user's question/request
            require_review: Force human review (None = auto-decide based on quality check)

        Returns:
            Final approved response
        """
        # Step 1: Create request
        request = Request(
            id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_query=user_query
        )
        print(f"\n{'='*80}")
        print(f"Processing Request: {request.id}")
        print(f"{'='*80}")

        # Step 2: Generate AI response
        print(f"\n Generating response using {self.config.generator_provider}...")
        response = self.generator.generate(request)
        print(f"   ✓ Response generated ({len(response.content)} chars)")

        # Step 3: Quality check
        print(f"\n️  Evaluating quality using {self.config.judge_provider}...")
        quality_check = self.checker.check(request, response)
        print(f"   ✓ Quality check complete")
        print(f"   Status: {'✅ PASSED' if quality_check.passed else '❌ FAILED'}")
        print(f"   Confidence: {quality_check.confidence:.2f}")
        if quality_check.reasoning:
            print(f"   Reasoning: {quality_check.reasoning}")

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
            print("\n️  Response EDITED")
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


def get_user_llm_choice() -> tuple[str, str, str, str]:
    """
    Prompt user to select generator LLM and automatically assign judge.

    Returns:
        Tuple of (generator_provider, generator_model, judge_provider, judge_model)
    """
    print("\n" + "=" * 80)
    print("SELECT LLM CONFIGURATION")
    print("=" * 80)
    print("\nWhich LLM should GENERATE responses?")
    print("  [1] Anthropic Claude (claude-sonnet-4-20250514)")
    print("  [2] OpenAI ChatGPT (gpt-4o-mini)")
    print("  [3] OpenAI ChatGPT (gpt-4o)")
    print("-" * 80)

    while True:
        choice = input("\nYour choice [1/2/3]: ").strip()

        if choice == "1":
            gen_provider = "anthropic"
            gen_model = "claude-sonnet-4-20250514"
            judge_provider = "openai"
            judge_model = "gpt-4o-mini"
            print(f"\n  Generator: Anthropic Claude")
            print(f"  Judge: OpenAI GPT-4o-mini (automatically assigned)")
            return gen_provider, gen_model, judge_provider, judge_model

        elif choice == "2":
            gen_provider = "openai"
            gen_model = "gpt-4o-mini"
            judge_provider = "anthropic"
            judge_model = "claude-sonnet-4-20250514"
            print(f"\n  Generator: OpenAI GPT-4o-mini")
            print(f"  Judge: Anthropic Claude (automatically assigned)")
            return gen_provider, gen_model, judge_provider, judge_model

        elif choice == "3":
            gen_provider = "openai"
            gen_model = "gpt-4o"
            judge_provider = "anthropic"
            judge_model = "claude-sonnet-4-20250514"
            print(f"\n  Generator: OpenAI GPT-4o")
            print(f"  Judge: Anthropic Claude (automatically assigned)")
            return gen_provider, gen_model, judge_provider, judge_model

        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


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
            # Ask if user wants to force review
            print("\nReview mode:")
            print("  [1] Auto (judge decides if review needed)")
            print("  [2] Force review for every response")
            review_choice = input("Choice [1/2]: ").strip()

            require_review = True if review_choice == "2" else None

            hitm_system.process(user_input, require_review=require_review)

        except KeyboardInterrupt:
            print("\n\n ️  Process interrupted. Returning to menu...")
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
    print("HUMAN IN THE MIDDLE - DUAL LLM SYSTEM")
    print("=" * 80)

    # Step 1: Locate and load environment file
    print("\n  Step 1: Locating environment file...")
    env_path = locating_environment_file()

    if env_path is None:
        print("\n  No environment file found!")
        print("\n  Please create a .env or keys.env file with:")
        print("ANTHROPIC_API_KEY=sk-ant-api03-your-key-here")
        print("OPENAI_API_KEY=sk-xxxxxx")
        sys.exit(1)

    # Step 2: Load API keys
    print("\n  Step 2: Loading API keys...")
    keys = load_api_keys(env_path)

    # Check for required keys
    anthropic_key = keys.get("ANTHROPIC_API_KEY")
    openai_key = keys.get("OPENAI_API_KEY")

    if not anthropic_key or not openai_key:
        print("\n  Both ANTHROPIC_API_KEY and OPENAI_API_KEY are required!")
        print("Please add them to your environment file.")
        sys.exit(1)

    # Step 3: Validate API keys
    print("\n  Step 3: Validating API keys...")
    if not validate_anthropic_api_key():
        sys.exit(1)

    if not validate_openai_api_key():
        sys.exit(1)

    # Step 4: Get user's LLM preference
    print("\n ️  Step 4: Configuring LLMs...")
    gen_provider, gen_model, judge_provider, judge_model = get_user_llm_choice()

    # Step 5: Create configuration
    print("\n ️  Step 5: Creating configuration...")
    config = HITMConfig(
        generator_provider=gen_provider,
        generator_model=gen_model,
        generator_temperature=0.7,
        generator_max_tokens=1024,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=0.3,
        judge_max_tokens=512,
        log_file="hitm_dual_llm_feedback.jsonl"
    )
    print("     Configuration created")

    # Step 6: Initialize HITM system
    print("\n  Step 6: Initializing HITM system...")
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

    # Step 7: Run interactive demo
    print("\n Starting interactive demo...")
    run_interactive_demo(hitm_system)


if __name__ == "__main__":
    main()
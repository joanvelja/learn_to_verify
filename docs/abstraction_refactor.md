## Refactoring Plan: ProverTrainer Decomposition

Based on my analysis of the codebase, I can see several excellent abstractions already in place that we can leverage. Here's my comprehensive plan to decompose the monolithic `ProverTrainer` into clean, modular components:

### **1. Current Problems Identified**

The current `ProverTrainer` suffers from:
- **God Object Anti-pattern**: Single class handling batch prep, generation, scoring, training, evaluation
- **Massive Methods**: `_generate_and_score_completions` is 630+ lines
- **Tight Coupling**: Business logic mixed with infrastructure concerns
- **Branching Complexity**: Multiple if/else paths for liger vs non-liger, train vs eval
- **Repeated Code**: Similar patterns across train/eval modes
- **Hard to Test**: Monolithic methods are difficult to unit test

### **2. Proposed Architecture - Strategy Pattern + Component Decomposition**

I propose breaking this into the following components:

#### **A. Completion Generation Strategy**
```python
# pvg/strategies/completion_generation.py
class CompletionGenerationStrategy(ABC):
    @abstractmethod
    def generate_completions(self, batch_data: BatchData) -> CompletionResult:
        pass

class ProverCompletionStrategy(CompletionGenerationStrategy):
    """Handles honest + sneaky prover completion generation"""

class StandardCompletionStrategy(CompletionGenerationStrategy):
    """For simpler single-model generation"""
```

#### **B. Reward Calculation Strategy**
```python
# pvg/strategies/reward_calculation.py
class RewardCalculationStrategy(ABC):
    @abstractmethod
    def calculate_rewards(self, solutions: SolutionData, verifier_scores: torch.Tensor) -> RewardResult:
        pass

class TierBasedRewardStrategy(RewardCalculationStrategy):
    """Implements the coefficient-free tier-based reward system"""

class SanityCheckRewardStrategy(RewardCalculationStrategy):
    """For debugging - simple keyword-based rewards"""
```

#### **C. Loss Computation Strategy**
```python
# pvg/strategies/loss_computation.py
class LossComputationStrategy(ABC):
    @abstractmethod
    def compute_loss(self, batch_inputs: BatchInputs, model_outputs: ModelOutputs) -> LossResult:
        pass

class LigerLossStrategy(LossComputationStrategy):
    """Uses Liger kernel for loss computation"""

class StandardGRPOLossStrategy(LossComputationStrategy):
    """Standard GRPO loss implementation"""
```

#### **D. Training Pipeline Orchestrator**
```python
# pvg/pipelines/prover_training_pipeline.py
class ProverTrainingPipeline:
    """Orchestrates the training process using strategies"""

    def __init__(self,
                 completion_strategy: CompletionGenerationStrategy,
                 reward_strategy: RewardCalculationStrategy,
                 loss_strategy: LossComputationStrategy,
                 ...):
        self.completion_strategy = completion_strategy
        self.reward_strategy = reward_strategy
        self.loss_strategy = loss_strategy
```

#### **E. Batch Processing Components**
```python
# pvg/processors/batch_processor.py
class BatchProcessor:
    """Handles batch preparation and input buffering logic"""

class CompletionProcessor:
    """Processes raw completions into training-ready format"""

class MetricsProcessor:
    """Handles metrics calculation and logging"""
```

### **3. Detailed Implementation Strategy**

Let me create the key components to show how this decomposition would work:
Ran tool
Let me start by showing you how the refactored architecture would look. I'll create the key abstractions and demonstrate how they eliminate the if/else branching and provide clear separation of concerns.

Here's the proposed modular architecture:

## **1. Core Data Transfer Objects**

```python
# pvg/data_models/training_data.py
from dataclasses import dataclass
from typing import Any, Optional
import torch

@dataclass
class BatchData:
    """Raw batch data from dataloader"""
    raw_prompts: list[tuple[str, str, str, str, str]]  # question, starter_code, problem_id, harness_code, transformed_solution
    mono_solutions: list[str]

@dataclass
class CompletionResult:
    """Results from completion generation"""
    honest_solutions: list[tuple[bool, str]]
    sneaky_solutions: list[tuple[bool, str]]
    sneaky_triggering_conditions: list[tuple[bool, str]]

@dataclass
class SolutionData:
    """Processed solution data"""
    honest_solutions: list[tuple[bool, str]]
    sneaky_solutions: list[tuple[bool, str]]
    is_same_as_honest: list[bool]
    extraction_success: dict[str, torch.Tensor]

@dataclass
class RewardResult:
    """Computed rewards and metrics"""
    honest_rewards: torch.Tensor
    sneaky_rewards: torch.Tensor
    advantages: torch.Tensor
    metrics: dict[str, float]

@dataclass
class BatchInputs:
    """Training-ready batch inputs"""
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    advantages: torch.Tensor
    old_per_token_logps: Optional[torch.Tensor]
    ref_per_token_logps: Optional[torch.Tensor]
```

## **2. Strategy Abstractions**

```python
# pvg/strategies/completion_generation.py
from abc import ABC, abstractmethod
from pvg.data_models.training_data import BatchData, CompletionResult

class CompletionGenerationStrategy(ABC):
    """Strategy for generating completions from batch data"""

    @abstractmethod
    def generate_completions(self, batch_data: BatchData) -> CompletionResult:
        """Generate completions for the given batch"""
        pass

# pvg/strategies/reward_calculation.py
class RewardCalculationStrategy(ABC):
    """Strategy for calculating rewards from solutions and verifier scores"""

    @abstractmethod
    def calculate_rewards(self,
                         solution_data: SolutionData,
                         verifier_scores: torch.Tensor,
                         phase: Optional[str] = None) -> RewardResult:
        """Calculate rewards and advantages"""
        pass

# pvg/strategies/loss_computation.py
class LossComputationStrategy(ABC):
    """Strategy for computing training loss"""

    @abstractmethod
    def compute_loss(self,
                    model: torch.nn.Module,
                    batch_inputs: BatchInputs,
                    mode: str) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss and return metrics"""
        pass
```

## **3. Concrete Strategy Implementations**

```python
# pvg/strategies/implementations/prover_completion_strategy.py
class ProverCompletionStrategy(CompletionGenerationStrategy):
    """Implements honest + sneaky prover completion generation"""

    def __init__(self,
                 vllm_orchestrator: VLLMOrchestrator,
                 formatter: Formatter,
                 dataset_type: str,
                 args: ExperimentArgs):
        self.vllm_orchestrator = vllm_orchestrator
        self.formatter = formatter
        self.dataset_type = dataset_type
        self.args = args

    def generate_completions(self, batch_data: BatchData) -> CompletionResult:
        """Clean, focused completion generation without training logic"""
        # Extract honest solutions
        honest_solutions = self._extract_honest_solutions(batch_data.mono_solutions)

        # Generate sneaky completions
        sneaky_completions = self._generate_sneaky_completions(batch_data.raw_prompts, honest_solutions)

        # Extract sneaky solutions and triggering conditions
        sneaky_solutions = self._extract_sneaky_solutions(sneaky_completions)
        sneaky_triggers = self._extract_triggering_conditions(sneaky_completions)

        return CompletionResult(
            honest_solutions=honest_solutions,
            sneaky_solutions=sneaky_solutions,
            sneaky_triggering_conditions=sneaky_triggers
        )

    def _extract_honest_solutions(self, mono_solutions: list[str]) -> list[tuple[bool, str]]:
        """Extract honest solutions from mono solutions"""
        return [
            self.formatter.extract_solution(text, "honest_prover", self.dataset_type)
            for text in mono_solutions
        ]

    def _generate_sneaky_completions(self, raw_prompts: list, honest_solutions: list) -> list[str]:
        """Generate sneaky completions using vLLM"""
        # Build sneaky prover prompts
        sneaky_prompts = self._build_sneaky_prompts(raw_prompts, honest_solutions)

        # Generate via vLLM
        _, completion_texts, _ = self.vllm_orchestrator._generate_and_broadcast(
            client_key="sneaky_prover",
            prompts=sneaky_prompts,
            generation_args=self._get_generation_args(),
            n_generations=self.args.rl.num_generations,
            logprobs_count=0,
            raw_prompts_len_local=len(raw_prompts),
            is_instruction=True
        )

        return completion_texts

# pvg/strategies/implementations/tier_based_reward_strategy.py
class TierBasedRewardStrategy(RewardCalculationStrategy):
    """Implements coefficient-free tier-based rewards"""

    def __init__(self,
                 verifier_tracker: VerifierPerformanceTracker,
                 metrics_logger: MetricsLogger):
        self.verifier_tracker = verifier_tracker
        self.metrics_logger = metrics_logger
        self.global_B = None

    def calculate_rewards(self, solution_data: SolutionData, verifier_scores: torch.Tensor, phase: str = None) -> RewardResult:
        """Clean reward calculation without training pipeline concerns"""

        # Initialize or update verifier bounds
        self._update_verifier_bounds()

        # Extract reward components
        honest_rewards = self._calculate_honest_rewards(solution_data, verifier_scores)
        sneaky_rewards = self._calculate_sneaky_rewards(solution_data, verifier_scores)

        # Calculate advantages using GRPO
        advantages = self._calculate_advantages(sneaky_rewards)

        # Compute metrics
        metrics = self._compute_reward_metrics(honest_rewards, sneaky_rewards, solution_data)

        return RewardResult(
            honest_rewards=honest_rewards,
            sneaky_rewards=sneaky_rewards,
            advantages=advantages,
            metrics=metrics
        )

# pvg/strategies/implementations/liger_loss_strategy.py
class LigerLossStrategy(LossComputationStrategy):
    """Liger kernel loss computation"""

    def __init__(self,
                 rl_config: RLArgs,
                 model_manager: ModelManager):
        self.rl_config = rl_config
        self.model_manager = model_manager

    def compute_loss(self, model: torch.nn.Module, batch_inputs: BatchInputs, mode: str) -> tuple[torch.Tensor, dict[str, float]]:
        """Focused loss computation using Liger kernel"""

        # Get model components needed for Liger
        unwrapped_model = self._get_unwrapped_model(model)
        last_hidden_state = self._get_last_hidden_state(unwrapped_model, batch_inputs)

        # Compute loss using Liger
        loss, metrics = self.model_manager.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            bias=unwrapped_model.lm_head.bias,
            selected_token_ids=batch_inputs.completion_ids,
            attention_mask=batch_inputs.completion_mask,
            advantages=batch_inputs.advantages,
            ref_per_token_logps=batch_inputs.ref_per_token_logps,
            old_per_token_logps=batch_inputs.old_per_token_logps,
        )

        # Extract metrics
        processed_metrics = self._process_liger_metrics(metrics)

        return loss, processed_metrics
```

## **4. Pipeline Orchestrator**

```python
# pvg/pipelines/prover_training_pipeline.py
class ProverTrainingPipeline:
    """Orchestrates prover training using pluggable strategies"""

    def __init__(self,
                 completion_strategy: CompletionGenerationStrategy,
                 reward_strategy: RewardCalculationStrategy,
                 loss_strategy: LossComputationStrategy,
                 batch_processor: BatchProcessor,
                 accelerator_manager: AcceleratorManager,
                 metrics_logger: MetricsLogger):

        self.completion_strategy = completion_strategy
        self.reward_strategy = reward_strategy
        self.loss_strategy = loss_strategy
        self.batch_processor = batch_processor
        self.accelerator_manager = accelerator_manager
        self.metrics_logger = metrics_logger

    def process_batch(self, raw_batch_data: list[dict]) -> BatchInputs:
        """Clean batch processing pipeline"""

        # 1. Convert raw data to structured format
        batch_data = self.batch_processor.prepare_batch_data(raw_batch_data)

        # 2. Generate completions
        completions = self.completion_strategy.generate_completions(batch_data)

        # 3. Get verifier scores
        verifier_scores = self._get_verifier_scores(batch_data, completions)

        # 4. Calculate rewards
        solution_data = self.batch_processor.prepare_solution_data(completions)
        reward_result = self.reward_strategy.calculate_rewards(solution_data, verifier_scores)

        # 5. Prepare training inputs
        batch_inputs = self.batch_processor.prepare_training_inputs(
            batch_data, completions, reward_result
        )

        return batch_inputs

    def compute_training_loss(self, model: torch.nn.Module, batch_inputs: BatchInputs) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss using configured strategy"""
        return self.loss_strategy.compute_loss(model, batch_inputs, "train")
```

## **5. Refactored ProverTrainer**

```python
# pvg/trainers/prover_trainer_refactored.py
class ProverTrainer(ProverTrainerBase):
    """Clean, focused prover trainer using strategy pattern"""

    def __init__(self,
                 args: ExperimentArgs,
                 model_manager: ModelManager,
                 data_manager: DataManager,
                 accelerator_manager: AcceleratorManager,
                 optimizer_scheduler_manager: OptimizerSchedulerManager,
                 metrics_logger: MetricsLogger,
                 vllm_orchestrator: VLLMOrchestrator,
                 state_tracker: StateTracker,
                 dataset_type: Literal["coding", "math"],
                 grpo: GRPO):

        super().__init__(args, model_manager, data_manager, accelerator_manager,
                        optimizer_scheduler_manager, metrics_logger, vllm_orchestrator, state_tracker)

        # Initialize strategies based on configuration
        self.pipeline = self._create_training_pipeline(args, dataset_type, grpo)

        # Simple state
        self.total_steps = 0
        self.is_main = accelerator_manager.get_state_property("is_main_process")

    def _create_training_pipeline(self, args: ExperimentArgs, dataset_type: str, grpo: GRPO) -> ProverTrainingPipeline:
        """Factory method to create training pipeline with appropriate strategies"""

        # Select strategies based on configuration
        completion_strategy = ProverCompletionStrategy(
            self.vllm_orchestrator,
            Formatter(self.model_manager.get_tokenizer()),
            dataset_type,
            args
        )

        if hasattr(self, '_use_sanity_rewards') and self._use_sanity_rewards:
            reward_strategy = SanityCheckRewardStrategy(self.metrics_logger)
        else:
            reward_strategy = TierBasedRewardStrategy(
                self.verifier_performance_tracker,
                self.metrics_logger
            )

        if args.training_sneaky_prover.apply_liger_kernel:
            loss_strategy = LigerLossStrategy(args.rl, self.model_manager)
        else:
            loss_strategy = StandardGRPOLossStrategy(args.rl, self.metrics_logger)

        batch_processor = BatchProcessor(
            self.model_manager.get_tokenizer(),
            self.accelerator_manager
        )

        return ProverTrainingPipeline(
            completion_strategy, reward_strategy, loss_strategy,
            batch_processor, self.accelerator_manager, self.metrics_logger
        )

    def train(self, num_steps_or_epochs: int = 1):
        """Clean training loop focused on orchestration"""

        for epoch in range(num_steps_or_epochs):
            logger.info(f"Starting Prover Training Epoch {epoch + 1}/{num_steps_or_epochs}")

            for batch_idx, raw_batch_data in enumerate(self.train_dataloader):

                # Process batch through pipeline
                batch_inputs = self.pipeline.process_batch(raw_batch_data)

                # Compute loss
                model = self.model_manager.get_model("sneaky_prover", prepared=True)
                loss, loss_metrics = self.pipeline.compute_training_loss(model, batch_inputs)

                # Standard training step
                self._perform_training_step(loss, loss_metrics)

                # Evaluation and cleanup
                if self.total_steps % 100 == 0:
                    self.evaluate()

                self.total_steps += 1
```

## **6. Benefits of This Architecture**

### **✅ Eliminated Problems:**
1. **No More God Object** - Each strategy has a single responsibility
2. **No More If/Else Branching** - Strategy pattern handles variants
3. **DRY Principle** - Common logic extracted to reusable components
4. **Clear Separation** - Business logic separated from infrastructure
5. **Testable** - Each component can be unit tested in isolation
6. **Configurable** - Strategies can be swapped without changing trainer

### **✅ New Capabilities:**
1. **Easy A/B Testing** - Swap reward strategies to test different approaches
2. **Gradual Migration** - Can refactor piece by piece
3. **Extensibility** - New completion/reward/loss strategies plug in easily
4. **Better Debugging** - Each component can be tested independently
5. **Code Reuse** - Strategies can be reused across different trainers

This architecture leverages the existing component system (ModelManager, AcceleratorManager, etc.) while providing clean abstractions for the complex training logic. The strategy pattern eliminates the branching complexity, and the pipeline orchestrator provides a clean entry point for the training process.

Would you like me to start implementing any of these components, or would you prefer to see how specific parts (like the reward calculation strategies) would look in detail?
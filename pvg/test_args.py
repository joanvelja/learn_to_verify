import unittest
import sys
from types import SimpleNamespace
from transformers import HfArgumentParser

# Assume pvg.utils is importable, adjust path if necessary
from pvg.utils import get_args, ModelConfig, DataConfig, TrainingConfig


class TestArgsParsing(unittest.TestCase):

    def test_nested_parsing_defaults(self):
        """Test if default values are loaded correctly into the nested structure."""
        # Simulate running with no command-line args
        original_argv = sys.argv
        sys.argv = [original_argv[0]]  # Just the script name

        args = get_args()

        # Check if the top-level structure is correct
        self.assertIsInstance(args, SimpleNamespace)
        self.assertTrue(hasattr(args, "model"))
        self.assertTrue(hasattr(args, "data"))
        self.assertTrue(hasattr(args, "training"))
        self.assertTrue(hasattr(args, "distributed"))
        self.assertTrue(hasattr(args, "inference"))
        self.assertTrue(hasattr(args, "logging"))
        self.assertTrue(hasattr(args, "instruction"))

        # Check if nested objects are the correct types
        self.assertIsInstance(args.model, ModelConfig)
        self.assertIsInstance(args.data, DataConfig)
        self.assertIsInstance(args.training, TrainingConfig)
        # ... etc. for all configs

        # Spot-check some default values
        self.assertEqual(args.model.honest_prover_name_or_path, "")
        self.assertEqual(args.training.learning_rate_a, 5e-5)
        self.assertEqual(args.logging.seed, 42)
        self.assertEqual(args.inference.vllm_host_a, "127.0.0.1")
        self.assertEqual(args.instruction.honest_prover_system_prompt, "")
        self.assertEqual(
            args.model.tokenizer_name_or_path, ""
        )  # Check default assignment logic

        # Restore sys.argv
        sys.argv = original_argv

    def test_nested_parsing_cmd_line(self):
        """Test parsing command-line arguments into the nested structure."""
        original_argv = sys.argv
        test_output_dir = "/tmp/test_output_nested"
        test_model_path = "test/model/a"
        test_lr = 1e-4
        test_seed = 1234
        # Simulate command-line arguments
        sys.argv = [
            original_argv[0],
            "--honest_prover_name_or_path",
            test_model_path,
            "--learning_rate_a",
            str(test_lr),
            "--output_dir",
            test_output_dir,
            "--seed",
            str(test_seed),
            "--vllm_port_a",
            "9999",
            "--train_num_samples",
            "5000",
        ]

        args = get_args()

        # Verify parsed values
        self.assertEqual(args.model.honest_prover_name_or_path, test_model_path)
        self.assertEqual(args.training.learning_rate_a, test_lr)
        self.assertEqual(args.logging.output_dir, test_output_dir)
        self.assertEqual(args.logging.seed, test_seed)
        self.assertEqual(args.inference.vllm_port_a, 9999)
        self.assertEqual(args.data.train_num_samples, 5000)

        # Check that unset arguments still have defaults
        self.assertEqual(args.training.learning_rate_b, 5e-5)
        self.assertEqual(args.model.sneaky_prover_name_or_path, "")

        # Check default assignment logic based on parsed value
        self.assertEqual(args.model.tokenizer_name_or_path, test_model_path)

        # Restore sys.argv
        sys.argv = original_argv

    def test_equivalence_to_flat(self):
        """
        (Conceptual Test)
        Ensures that for the same command-line input, the nested structure
        holds the same values as the (old) flat structure would have.
        This requires temporarily keeping the FlatExperimentArgs and its parsing logic
        or having a pre-recorded set of outputs from the flat version.
        """
        # 1. Define a set of command-line arguments
        cmd_line_args = [
            "--honest_prover_name_or_path",
            "path/a",
            "--sneaky_prover_name_or_path",
            "path/b",
            "--dataset_name",
            "my_data",
            "--learning_rate_a",
            "1e-4",
            "--per_device_train_batch_size",
            "8",
            "--ds_config_honest_prover",
            "ds_a.json",
            "--vllm_port_c",
            "8005",
            "--output_dir",
            "/path/to/output",
            "--seed",
            "99",
            # ... add more representative args
        ]

        # 2. Parse using the NEW nested structure
        original_argv = sys.argv
        sys.argv = [original_argv[0]] + cmd_line_args
        nested_args = get_args()
        sys.argv = original_argv  # Restore

        # 3. Parse using the OLD flat structure (assuming you keep it temporarily)
        from pvg.utils import FlatExperimentArgs  # Temporarily import old class

        parser_flat = HfArgumentParser((FlatExperimentArgs,))
        sys.argv = [original_argv[0]] + cmd_line_args
        flat_args = parser_flat.parse_args_into_dataclasses()[0]
        sys.argv = original_argv  # Restore

        # --- OR ---
        # 3b. Load pre-recorded expected values for the flat structure with these inputs
        # expected_flat_values = {
        #    'honest_prover_name_or_path': 'path/a',
        #    'sneaky_prover_name_or_path': 'path/b',
        #    'dataset_name': 'my_data',
        #    'learning_rate_a': 1e-4,
        #    'per_device_train_batch_size': 8,
        #    'ds_config_honest_prover': 'ds_a.json',
        #    'vllm_port_c': 8005,
        #    'output_dir': '/path/to/output',
        #    'seed': 99,
        #    # ... etc
        # }

        # 4. Assert equivalence for all corresponding fields
        self.assertEqual(
            nested_args.model.honest_prover_name_or_path,
            flat_args.honest_prover_name_or_path,
        )
        self.assertEqual(
            nested_args.model.sneaky_prover_name_or_path,
            flat_args.sneaky_prover_name_or_path,
        )
        self.assertEqual(nested_args.data.dataset_name, flat_args.dataset_name)
        self.assertEqual(
            nested_args.training.learning_rate_a, flat_args.learning_rate_a
        )
        self.assertEqual(
            nested_args.training.per_device_train_batch_size,
            flat_args.per_device_train_batch_size,
        )
        self.assertEqual(
            nested_args.distributed.ds_config_honest_prover,
            flat_args.ds_config_honest_prover,
        )
        self.assertEqual(nested_args.inference.vllm_port_c, flat_args.vllm_port_c)
        self.assertEqual(nested_args.logging.output_dir, flat_args.output_dir)
        self.assertEqual(nested_args.logging.seed, flat_args.seed)
        # ... etc. for all arguments

        # Using the dictionary approach:
        # self.assertEqual(nested_args.model.honest_prover_name_or_path, expected_flat_values['honest_prover_name_or_path'])
        # ... etc.

        # Placeholder assertion for the conceptual test
        self.assertTrue(True, "Conceptual equivalence check passed (implement fully)")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

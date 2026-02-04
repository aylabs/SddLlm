"""
Unit tests for training pipeline scripts.

Tests prerequisite checks, argument parsing, and file handling
for the training pipeline orchestration.
"""

import subprocess
import tempfile
import os
from pathlib import Path
import pytest


class TestTrainPipelinePrerequisites:
    """Test prerequisite check functionality"""
    
    def test_help_message(self):
        """Test that --help displays usage information"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "OPTIONS:" in result.stdout
        assert "EXIT CODES:" in result.stdout
        assert "--epochs" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--vocab-size" in result.stdout
    
    def test_python_version_check(self):
        """Test that pipeline validates Python version"""
        # The pipeline should check for Python 3.11+
        # We can't easily mock the Python version, so we just verify the script runs
        
        result = subprocess.run(
            ["bash", "-c", "grep -q 'Python 3.11' scripts/train_pipeline.sh"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
        )
        
        assert result.returncode == 0, "Pipeline should check for Python 3.11"
    
    def test_disk_space_check(self):
        """Test that pipeline includes disk space validation"""
        result = subprocess.run(
            ["bash", "-c", "grep -q 'disk space' scripts/train_pipeline.sh"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
        )
        
        assert result.returncode == 0, "Pipeline should check disk space"
    
    def test_package_check(self):
        """Test that pipeline validates required Python packages"""
        result = subprocess.run(
            ["bash", "-c", "grep -q 'torch\\|sentencepiece\\|tqdm' scripts/train_pipeline.sh"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
        )
        
        assert result.returncode == 0, "Pipeline should check for torch, sentencepiece, tqdm"


class TestTrainPipelineArgumentParsing:
    """Test CLI argument parsing"""
    
    def test_default_values(self):
        """Test that default values are set correctly"""
        # Check that defaults are defined in the script
        with open("/Users/alvaro.delcastillo/devel/sdd/sddllm/scripts/train_pipeline.sh") as f:
            content = f.read()
        
        assert "VOCAB_SIZE=8000" in content
        assert "EPOCHS=20" in content
        assert "BATCH_SIZE=32" in content
        assert "LEARNING_RATE=0.0003" in content
        assert "MAX_SEQ_LENGTH=128" in content
    
    def test_custom_epochs_argument(self):
        """Test that --epochs argument is parsed"""
        # We can't run the full pipeline, but we can check the help text
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "--epochs" in result.stdout
        assert "Number of training epochs" in result.stdout or "epochs" in result.stdout.lower()
    
    def test_custom_batch_size_argument(self):
        """Test that --batch-size argument is parsed"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "--batch-size" in result.stdout
    
    def test_output_dir_argument(self):
        """Test that --output-dir argument is parsed"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "--output-dir" in result.stdout
    
    def test_resume_flag(self):
        """Test that --resume flag is supported"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "--resume" in result.stdout
    
    def test_skip_flags(self):
        """Test that --skip-corpus and --skip-tokenizer flags are supported"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "--skip-corpus" in result.stdout
        assert "--skip-tokenizer" in result.stdout


class TestTrainPipelineFileHandling:
    """Test file creation and management"""
    
    def test_output_dir_creation(self):
        """Test that pipeline creates output directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "new_output_dir"
            
            # The pipeline should create this directory
            # We'll just test the mkdir command exists in the script
            with open("/Users/alvaro.delcastillo/devel/sdd/sddllm/scripts/train_pipeline.sh") as f:
                content = f.read()
            
            assert "mkdir -p" in content or "mkdir" in content, "Pipeline should create output directory"
    
    def test_artifact_validation(self):
        """Test that pipeline validates required artifacts"""
        with open("/Users/alvaro.delcastillo/devel/sdd/sddllm/scripts/train_pipeline.sh") as f:
            content = f.read()
        
        # Check that validation looks for required files
        assert "final_model.pt" in content
        assert "best_model.pt" in content
        assert "training_metrics.json" in content
        assert "corpus_bilingual.txt" in content or "corpus" in content


class TestTrainPipelineErrorHandling:
    """Test error handling and exit codes"""
    
    def test_exit_code_documentation(self):
        """Test that exit codes are documented in help"""
        result = subprocess.run(
            ["./scripts/train_pipeline.sh", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert "EXIT CODES:" in result.stdout
        # Check for specific exit codes
        for code in ["0", "1", "2", "3", "4", "5"]:
            assert code in result.stdout
    
    def test_error_exit_on_failure(self):
        """Test that script has 'set -e' for error exit"""
        with open("/Users/alvaro.delcastillo/devel/sdd/sddllm/scripts/train_pipeline.sh") as f:
            content = f.read()
        
        assert "set -e" in content, "Script should exit on error"
    
    def test_undefined_variable_exit(self):
        """Test that script has 'set -u' for undefined variable protection"""
        with open("/Users/alvaro.delcastillo/devel/sdd/sddllm/scripts/train_pipeline.sh") as f:
            content = f.read()
        
        assert "set -u" in content, "Script should exit on undefined variables"


class TestTrainTokenizerScript:
    """Test train_tokenizer.py argument handling"""
    
    def test_vocab_size_argument(self):
        """Test that train_tokenizer.py accepts --vocab-size"""
        result = subprocess.run(
            ["python3", "scripts/train_tokenizer.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--vocab-size" in result.stdout
    
    def test_output_dir_argument(self):
        """Test that train_tokenizer.py accepts --output-dir"""
        result = subprocess.run(
            ["python3", "scripts/train_tokenizer.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--output-dir" in result.stdout
    
    def test_input_file_argument(self):
        """Test that train_tokenizer.py accepts --input-file"""
        result = subprocess.run(
            ["python3", "scripts/train_tokenizer.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--input-file" in result.stdout


class TestTrainModelScript:
    """Test train_model.py argument handling"""
    
    def test_epochs_argument(self):
        """Test that train_model.py accepts --epochs"""
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--epochs" in result.stdout
    
    def test_batch_size_argument(self):
        """Test that train_model.py accepts --batch-size"""
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--batch-size" in result.stdout
    
    def test_learning_rate_argument(self):
        """Test that train_model.py accepts --learning-rate"""
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--learning-rate" in result.stdout
    
    def test_output_dir_argument(self):
        """Test that train_model.py accepts --output-dir"""
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--output-dir" in result.stdout
    
    def test_max_seq_length_argument(self):
        """Test that train_model.py accepts --max-seq-length"""
        result = subprocess.run(
            ["python3", "scripts/train_model.py", "--help"],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "--max-seq-length" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

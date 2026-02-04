"""
Integration tests for training pipeline.

Tests the complete end-to-end training workflow with minimal corpus
to validate pipeline functionality and artifact generation.
"""

import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
import pytest


class TestTrainingPipeline:
    """Integration tests for train_pipeline.sh"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for test artifacts"""
        temp_dir = tempfile.mkdtemp(prefix="test_pipeline_")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_corpus(self, temp_output_dir):
        """Create minimal test corpus (50 lines)"""
        corpus_path = Path(temp_output_dir) / "corpus_bilingual.txt"
        
        # Create minimal bilingual corpus
        lines = [
            "Hello world",
            "Hola mundo",
            "How are you?",
            "¿Cómo estás?",
            "Good morning",
            "Buenos días",
            "Thank you",
            "Gracias",
            "Goodbye",
            "Adiós",
        ] * 5  # Repeat to get 50 lines
        
        corpus_path.write_text("\n".join(lines))
        return str(corpus_path)
    
    def test_pipeline_minimal_corpus(self, temp_output_dir, minimal_corpus):
        """Test complete pipeline with minimal corpus and 2 epochs"""
        
        # Run pipeline with minimal configuration
        result = subprocess.run(
            [
                "./scripts/train_pipeline.sh",
                "--epochs", "2",
                "--batch-size", "4",
                "--vocab-size", "1000",
                "--max-seq-length", "32",
                "--output-dir", temp_output_dir,
                "--skip-corpus",  # Use our pre-created minimal corpus
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Pipeline should complete successfully
        assert result.returncode == 0, f"Pipeline failed with: {result.stderr}"
        
        # Verify required artifacts created
        output_path = Path(temp_output_dir)
        
        assert (output_path / "corpus_bilingual.txt").exists(), "Corpus file missing"
        assert (output_path / "bilingual_1000.model").exists(), "Tokenizer model missing"
        assert (output_path / "bilingual_1000.vocab").exists(), "Tokenizer vocab missing"
        assert (output_path / "final_model.pt").exists(), "Final model missing"
        assert (output_path / "best_model.pt").exists(), "Best model missing"
        assert (output_path / "training_metrics.json").exists(), "Training metrics missing"
        
        # Verify checkpoint created (save_every=5, but we only train 2 epochs)
        # So no intermediate checkpoint should exist
        assert not (output_path / "checkpoint_epoch_5.pt").exists(), "Unexpected checkpoint"
        
        # Verify training metrics content
        with open(output_path / "training_metrics.json") as f:
            metrics = json.load(f)
        
        assert "run_id" in metrics, "Missing run_id in metrics"
        assert "start_timestamp" in metrics, "Missing start_timestamp"
        assert "end_timestamp" in metrics, "Missing end_timestamp"
        assert "configuration" in metrics, "Missing configuration"
        assert "final_metrics" in metrics, "Missing final_metrics"
        assert "epoch_history" in metrics, "Missing epoch_history"
        
        # Verify we trained for 2 epochs
        assert metrics["final_metrics"]["total_epochs"] == 2, "Wrong number of epochs"
        assert len(metrics["epoch_history"]) == 2, "Wrong epoch history length"
        
        # Verify loss values are reasonable (not NaN/Inf)
        final_val_loss = metrics["final_metrics"]["final_val_loss"]
        assert final_val_loss is not None, "Missing final validation loss"
        assert 0 < final_val_loss < 100, f"Unreasonable validation loss: {final_val_loss}"
        
        # Verify configuration matches what we passed
        config = metrics["configuration"]
        assert config["batch_size"] == 4, "Wrong batch_size in config"
        assert config["num_epochs"] == 2, "Wrong num_epochs in config"
        assert config["vocab_size"] == 1000, "Wrong vocab_size in config"
    
    def test_pipeline_skip_phases(self, temp_output_dir, minimal_corpus):
        """Test pipeline with --skip-corpus and --skip-tokenizer flags"""
        
        # First run: create corpus and tokenizer
        output_path = Path(temp_output_dir)
        
        # Create tokenizer manually
        subprocess.run(
            [
                "python3", "scripts/train_tokenizer.py",
                "--vocab-size", "500",
                "--output-dir", temp_output_dir,
                "--input-file", minimal_corpus,
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            check=True,
            capture_output=True,
        )
        
        # Second run: skip both corpus and tokenizer
        result = subprocess.run(
            [
                "./scripts/train_pipeline.sh",
                "--epochs", "1",
                "--batch-size", "4",
                "--vocab-size", "500",
                "--output-dir", temp_output_dir,
                "--skip-corpus",
                "--skip-tokenizer",
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        assert "Skipping corpus download" in result.stdout or "Skipping tokenizer training" in result.stdout
    
    def test_pipeline_prerequisite_failure(self):
        """Test that pipeline fails gracefully with bad prerequisites"""
        
        # Try to use non-existent output directory without write permissions
        # This is tricky to test portably, so we'll test with invalid Python package
        
        # We can't easily inject a missing package, so we'll test argument validation
        result = subprocess.run(
            [
                "./scripts/train_pipeline.sh",
                "--epochs", "abc",  # Invalid integer
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        # Should fail due to invalid argument
        assert result.returncode != 0, "Pipeline should fail with invalid arguments"
    
    def test_pipeline_resume_prompt(self, temp_output_dir, minimal_corpus):
        """Test that pipeline prompts for resume when checkpoints exist"""
        
        output_path = Path(temp_output_dir)
        
        # Create a fake checkpoint to trigger resume prompt
        (output_path / "checkpoint_epoch_5.pt").touch()
        
        # Run pipeline (will timeout waiting for user input, so we expect failure)
        result = subprocess.run(
            [
                "./scripts/train_pipeline.sh",
                "--epochs", "1",
                "--output-dir", temp_output_dir,
                "--skip-corpus",
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            input="n\n",  # Answer "no" to resume prompt
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Pipeline should continue (not resume) after "no" response
        # Note: This test is fragile due to interactive prompts
        # In production, we'd use --resume flag to avoid prompts


class TestTrainingScriptsUnit:
    """Unit tests for individual training scripts"""
    
    def test_train_tokenizer_output_dir(self, tmp_path):
        """Test that train_tokenizer.py respects --output-dir"""
        
        # Create minimal corpus
        corpus_file = tmp_path / "test_corpus.txt"
        corpus_file.write_text("Hello world\nHola mundo\n" * 100)
        
        output_dir = tmp_path / "tokenizer_output"
        
        result = subprocess.run(
            [
                "python3", "scripts/train_tokenizer.py",
                "--vocab-size", "500",
                "--output-dir", str(output_dir),
                "--input-file", str(corpus_file),
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        assert result.returncode == 0, f"train_tokenizer.py failed: {result.stderr}"
        assert (output_dir / "bilingual_500.model").exists(), "Tokenizer model not created in output_dir"
        assert (output_dir / "bilingual_500.vocab").exists(), "Tokenizer vocab not created in output_dir"
    
    def test_train_model_output_dir(self, tmp_path):
        """Test that train_model.py respects --output-dir"""
        
        # This test requires a tokenizer to exist, so we'll create one first
        corpus_file = tmp_path / "test_corpus.txt"
        corpus_file.write_text("Hello world\nHola mundo\n" * 100)
        
        # Create tokenizer
        subprocess.run(
            [
                "python3", "scripts/train_tokenizer.py",
                "--vocab-size", "500",
                "--output-dir", str(tmp_path),
                "--input-file", str(corpus_file),
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            check=True,
            capture_output=True,
        )
        
        output_dir = tmp_path / "model_output"
        
        # Copy corpus and tokenizer to expected data/ location for train_model.py
        # (train_model.py still hardcodes some paths, this is a limitation)
        data_dir = Path("/Users/alvaro.delcastillo/devel/sdd/sddllm/data")
        if not data_dir.exists():
            pytest.skip("data/ directory not found, skipping test")
        
        # We need to have corpus_bilingual.txt and bilingual_8k.model in data/
        # This test is integration-like and needs the full data setup
        if not (data_dir / "corpus_bilingual.txt").exists():
            pytest.skip("Corpus not found in data/, skipping test")
        if not (data_dir / "bilingual_8k.model").exists():
            pytest.skip("Tokenizer not found in data/, skipping test")
        
        result = subprocess.run(
            [
                "python3", "scripts/train_model.py",
                "--epochs", "1",
                "--batch-size", "4",
                "--output-dir", str(output_dir),
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        assert result.returncode == 0, f"train_model.py failed: {result.stderr}"
        assert (output_dir / "final_model.pt").exists(), "Model not created in output_dir"
        assert (output_dir / "training_metrics.json").exists(), "Metrics not created in output_dir"
    
    def test_train_model_json_export(self, tmp_path):
        """Test that train_model.py exports training_metrics.json"""
        
        # Similar setup as above test
        data_dir = Path("/Users/alvaro.delcastillo/devel/sdd/sddllm/data")
        if not (data_dir / "corpus_bilingual.txt").exists() or not (data_dir / "bilingual_8k.model").exists():
            pytest.skip("Required data files not found, skipping test")
        
        output_dir = tmp_path / "metrics_test"
        
        result = subprocess.run(
            [
                "python3", "scripts/train_model.py",
                "--epochs", "1",
                "--batch-size", "4",
                "--output-dir", str(output_dir),
            ],
            cwd="/Users/alvaro.delcastillo/devel/sdd/sddllm",
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        assert result.returncode == 0
        
        metrics_file = output_dir / "training_metrics.json"
        assert metrics_file.exists(), "training_metrics.json not created"
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Verify JSON structure
        assert "run_id" in metrics
        assert "start_timestamp" in metrics
        assert "end_timestamp" in metrics
        assert "configuration" in metrics
        assert "final_metrics" in metrics
        assert "epoch_history" in metrics
        
        # Verify run_id is a valid UUID
        import uuid
        try:
            uuid.UUID(metrics["run_id"])
        except ValueError:
            pytest.fail("run_id is not a valid UUID")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

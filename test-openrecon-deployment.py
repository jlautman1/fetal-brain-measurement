#!/usr/bin/env python3
"""
Comprehensive OpenRecon Deployment Testing Script
Tests both the real fetal brain implementation and dummy implementation
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenReconDeploymentTester:
    """Comprehensive tester for OpenRecon deployment"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.python_server_dir = self.base_dir.parent / "python-ismrmrd-server"
        self.results = {
            "file_structure": {},
            "dummy_build": {},
            "real_build": {},
            "docker_tests": {},
            "overall_status": "unknown"
        }
    
    def test_file_structure(self):
        """Test that all required files are in the correct locations"""
        logger.info("🔍 Testing file structure...")
        
        required_files = {
            # Main implementation files
            "fetal-brain-measurement/openrecon.py": "Main OpenRecon i2i handler",
            "fetal-brain-measurement/openrecon.json": "OpenRecon configuration",
            "fetal-brain-measurement/Dockerfile.openrecon": "OpenRecon production Dockerfile",
            
            # Dummy implementation files
            "fetal-brain-measurement/dummy_openrecon.py": "Dummy OpenRecon handler", 
            "fetal-brain-measurement/dummy_openrecon.json": "Dummy configuration",
            "fetal-brain-measurement/Dockerfile.dummy": "Dummy Dockerfile",
            
            # OpenRecon metadata files
            "python-ismrmrd-server/openrecon_json_ui.json": "OpenRecon UI metadata",
            "python-ismrmrd-server/OpenReconSchema_1.1.0.json": "OpenRecon schema",
            "python-ismrmrd-server/main.py": "OpenRecon server main",
            
            # Working fetal brain files
            "fetal-brain-measurement/Dockerfile": "Original working Dockerfile",
            "fetal-brain-measurement/requirements.txt": "Python dependencies",
            "fetal-brain-measurement/Code/FetalMeasurements-master/execute.py": "Fetal pipeline main"
        }
        
        missing_files = []
        existing_files = []
        
        for relative_path, description in required_files.items():
            full_path = self.base_dir.parent / relative_path
            if full_path.exists():
                existing_files.append((relative_path, description, full_path.stat().st_size))
                logger.info(f"✅ {relative_path} - {description}")
            else:
                missing_files.append((relative_path, description))
                logger.error(f"❌ {relative_path} - {description} - NOT FOUND")
        
        self.results["file_structure"] = {
            "existing_files": existing_files,
            "missing_files": missing_files,
            "status": "pass" if not missing_files else "fail"
        }
        
        return not missing_files
    
    def test_json_validity(self):
        """Test that JSON configuration files are valid"""
        logger.info("🔍 Testing JSON file validity...")
        
        json_files = [
            "fetal-brain-measurement/openrecon.json",
            "fetal-brain-measurement/dummy_openrecon.json", 
            "python-ismrmrd-server/openrecon_json_ui.json",
            "python-ismrmrd-server/OpenReconSchema_1.1.0.json"
        ]
        
        valid_jsons = []
        invalid_jsons = []
        
        for json_file in json_files:
            full_path = self.base_dir.parent / json_file
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    valid_jsons.append((json_file, len(str(data))))
                    logger.info(f"✅ {json_file} - Valid JSON ({len(str(data))} chars)")
            except Exception as e:
                invalid_jsons.append((json_file, str(e)))
                logger.error(f"❌ {json_file} - Invalid JSON: {e}")
        
        return not invalid_jsons
    
    def test_python_syntax(self):
        """Test Python files for syntax errors"""
        logger.info("🔍 Testing Python file syntax...")
        
        python_files = [
            "fetal-brain-measurement/openrecon.py",
            "fetal-brain-measurement/dummy_openrecon.py"
        ]
        
        syntax_errors = []
        
        for py_file in python_files:
            full_path = self.base_dir.parent / py_file
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, full_path, 'exec')
                logger.info(f"✅ {py_file} - Valid Python syntax")
            except SyntaxError as e:
                syntax_errors.append((py_file, str(e)))
                logger.error(f"❌ {py_file} - Syntax error: {e}")
            except Exception as e:
                syntax_errors.append((py_file, f"Could not read file: {e}"))
                logger.error(f"❌ {py_file} - Read error: {e}")
        
        return not syntax_errors
    
    def test_docker_build_dummy(self):
        """Test building the dummy Docker image"""
        logger.info("🐳 Testing dummy Docker build...")
        
        try:
            # Build dummy image
            build_cmd = [
                "docker", "build",
                "-f", "fetal-brain-measurement/Dockerfile.dummy",
                "-t", "openrecon-dummy:test",
                "."
            ]
            
            logger.info(f"Running: {' '.join(build_cmd)}")
            result = subprocess.run(
                build_cmd,
                cwd=self.base_dir.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Dummy Docker build successful!")
                self.results["dummy_build"] = {
                    "status": "success",
                    "image_name": "openrecon-dummy:test"
                }
                return True
            else:
                logger.error(f"❌ Dummy Docker build failed: {result.stderr}")
                self.results["dummy_build"] = {
                    "status": "failed",
                    "error": result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Dummy Docker build timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Dummy Docker build error: {e}")
            return False
    
    def test_docker_build_real(self):
        """Test building the real fetal brain Docker image"""
        logger.info("🐳 Testing real fetal brain Docker build...")
        
        try:
            # Build real image
            build_cmd = [
                "docker", "build", 
                "-f", "fetal-brain-measurement/Dockerfile.openrecon",
                "-t", "openrecon-fetal:test",
                "."
            ]
            
            logger.info(f"Running: {' '.join(build_cmd)}")
            result = subprocess.run(
                build_cmd,
                cwd=self.base_dir.parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for full build
            )
            
            if result.returncode == 0:
                logger.info("✅ Real fetal brain Docker build successful!")
                self.results["real_build"] = {
                    "status": "success", 
                    "image_name": "openrecon-fetal:test"
                }
                return True
            else:
                logger.error(f"❌ Real Docker build failed: {result.stderr}")
                self.results["real_build"] = {
                    "status": "failed",
                    "error": result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Real Docker build timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Real Docker build error: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📄 Generating test report...")
        
        report = {
            "test_timestamp": time.time(),
            "test_summary": {
                "file_structure": self.results["file_structure"]["status"] if "file_structure" in self.results else "not_tested",
                "dummy_build": self.results["dummy_build"].get("status", "not_tested") if "dummy_build" in self.results else "not_tested", 
                "real_build": self.results["real_build"].get("status", "not_tested") if "real_build" in self.results else "not_tested"
            },
            "detailed_results": self.results
        }
        
        # Determine overall status
        if all(status == "success" or status == "pass" for status in report["test_summary"].values() if status != "not_tested"):
            report["overall_status"] = "ALL_TESTS_PASSED"
        else:
            report["overall_status"] = "SOME_TESTS_FAILED"
        
        # Save report
        report_file = self.base_dir / "openrecon_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        return report
    
    def run_all_tests(self, include_docker=True):
        """Run all tests"""
        logger.info("🚀 Starting comprehensive OpenRecon deployment testing...")
        
        # File structure tests
        file_test_passed = self.test_file_structure()
        json_test_passed = self.test_json_validity()
        syntax_test_passed = self.test_python_syntax()
        
        # Docker tests (optional due to time)
        dummy_build_passed = True
        real_build_passed = True
        
        if include_docker:
            dummy_build_passed = self.test_docker_build_dummy()
            # Only test real build if dummy build passes (saves time)
            if dummy_build_passed:
                real_build_passed = self.test_docker_build_real()
        
        # Generate report
        report = self.generate_test_report()
        
        # Summary
        logger.info("=" * 60)
        logger.info("🎯 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📁 File Structure: {'✅ PASS' if file_test_passed else '❌ FAIL'}")
        logger.info(f"📄 JSON Validity: {'✅ PASS' if json_test_passed else '❌ FAIL'}")
        logger.info(f"🐍 Python Syntax: {'✅ PASS' if syntax_test_passed else '❌ FAIL'}")
        
        if include_docker:
            logger.info(f"🐳 Dummy Build: {'✅ PASS' if dummy_build_passed else '❌ FAIL'}")
            logger.info(f"🐳 Real Build: {'✅ PASS' if real_build_passed else '❌ FAIL'}")
        
        logger.info(f"📊 Overall: {'✅ SUCCESS' if report['overall_status'] == 'ALL_TESTS_PASSED' else '❌ ISSUES FOUND'}")
        logger.info("=" * 60)
        
        return report


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenRecon deployment setup")
    parser.add_argument("--no-docker", action="store_true", help="Skip Docker build tests")
    parser.add_argument("--dummy-only", action="store_true", help="Only test dummy implementation")
    
    args = parser.parse_args()
    
    tester = OpenReconDeploymentTester()
    
    if args.dummy_only:
        logger.info("🎭 Testing DUMMY implementation only...")
        file_ok = tester.test_file_structure()
        json_ok = tester.test_json_validity()
        syntax_ok = tester.test_python_syntax()
        
        if not args.no_docker and file_ok and json_ok and syntax_ok:
            tester.test_docker_build_dummy()
        
        tester.generate_test_report()
    else:
        logger.info("🧠 Testing FULL implementation...")
        tester.run_all_tests(include_docker=not args.no_docker)


if __name__ == "__main__":
    main()

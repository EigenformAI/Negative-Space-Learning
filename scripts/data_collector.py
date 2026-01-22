import subprocess
import sys
import json
import hashlib
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from loguru import logger
import docker

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CompleteDataCollector:
    def __init__(self, target_files: int = 20000, completions_per_prompt: int = 1):
        self.target_files = target_files
        self.completions_per_prompt = completions_per_prompt
        self.docker_client = docker.from_env()
        
        # Container IDs (with hyphens as shown in debug output)
        self.container_ids = [
            "special-learn-compose_service-a_1",
            "special-learn-compose_service-b_1"
        ]
        
        # Auto-detect container names if the hardcoded ones don't exist
        try:
            self.docker_client.containers.get(self.container_ids[0])
            logger.info(f"✅ Using configured container names: {self.container_ids}")
        except docker.errors.NotFound:
            logger.warning("Configured container names not found, auto-detecting...")
            if not self.update_container_names():
                logger.error("Failed to find containers. Please ensure they are running.")
                logger.info("Start containers with: cd docker/special-learn-compose && docker-compose up -d")
                raise Exception("Containers not found")
        
        # Data paths (customize as needed)
        self.base_folder = Path("./data/Your-Model-Name/multi-container")
        self.data_folder = Path("./data/Your-Model-Name/multi-container/train_data")
        self.grouped_folder = Path("./data/Your-Model-Name/multi-container/grouped_completions")
        self.metrics_folder = Path("./data/Your-Model-Name/multi-container/metrics")
        
        # Create all folders
        for folder in [self.data_folder, self.grouped_folder, self.metrics_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.iteration = 0
        self.prompt_registry = {}  # prompt_hash -> prompt info
        self.prompt_groups = defaultdict(list)  # prompt_hash -> list of completions
        self.collection_stats = []
        self.restart_interval = 10  # Restart containers every 10 iterations
        
        # Setup logging
        logger.add(self.metrics_folder / "collection.log", rotation="100 MB")
    
    def update_container_names(self):
        """Dynamically find and update container names."""
        logger.info("🔍 Finding current container names...")
        
        containers = self.docker_client.containers.list()
        found_containers = []
        
        for container in containers:
            if "special-learn-compose" in container.name and "service" in container.name:
                found_containers.append(container.name)
                logger.info(f"  Found: {container.name}")
        
        if len(found_containers) >= 2:
            # Sort to ensure consistent order
            found_containers.sort()
            # Ensure service-a comes before service-b
            if "service-b" in found_containers[0] and "service-a" in found_containers[1]:
                found_containers = [found_containers[1], found_containers[0]]
            
            self.container_ids = found_containers[:2]  # Take first 2 if more exist
            logger.info(f"✅ Updated container IDs: {self.container_ids}")
            return True
        else:
            logger.error(f"❌ Expected 2 containers, found {len(found_containers)}")
            return False
    
    def get_mixed_cleanup_variations(self) -> List[Dict]:
        """
        Get MIXED cleanup variations - each variation contains ALL types of files.
        This is more realistic and provides better training diversity.
        """
        return [
            {
                "name": "variation_1_heavy",
                "commands": [
                    # Create all directories
                    "mkdir -p /tmp/cleanup /tmp/sessions /var/log/app /var/cache/pkg /tmp/db_temp /home/alice/trash /var/tmp/build",
                    # Temp files (40 files, 4MB)
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/tmp/cleanup/temp_$i.tmp bs=100K count=1 2>/dev/null; done",
                    # Log files (30 files, 3MB)
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/var/log/app/app_$i.log bs=100K count=1 2>/dev/null; done",
                    # Cache files (25 files, 2.5MB)
                    "for i in $(seq 1 25); do dd if=/dev/zero of=/var/cache/pkg/cache_$i.dat bs=100K count=1 2>/dev/null; done",
                    # Database temp (15 files, 3MB)
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/db_temp/table_$i.tmp bs=200K count=1 2>/dev/null; done",
                    # User trash (20 files, 1MB)
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/home/alice/trash/deleted_$i.bak bs=50K count=1 2>/dev/null; done",
                    # Build artifacts (10 files, 2MB)
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/build/obj_$i.o bs=200K count=1 2>/dev/null; done"
                ],
                "expected_kb": 15500,
                "description": "Heavy mix - lots of large files across all categories"
            },
            {
                "name": "variation_2_medium",
                "commands": [
                    # Create directories
                    "mkdir -p /tmp/work /var/log/system /var/cache/app /tmp/downloads /home/alice/.cache /var/tmp/sql",
                    # Temp files (50 files, 2.5MB)
                    "for i in $(seq 1 50); do dd if=/dev/zero of=/tmp/work/work_$i.tmp bs=50K count=1 2>/dev/null; done",
                    # Log files (40 files, 2MB)
                    "for i in $(seq 1 40); do dd if=/dev/zero of=/var/log/system/sys_$i.log bs=50K count=1 2>/dev/null; done",
                    # Cache files (35 files, 2.625MB)
                    "for i in $(seq 1 35); do dd if=/dev/zero of=/var/cache/app/app_$i.cache bs=75K count=1 2>/dev/null; done",
                    # Downloads (15 files, 2.25MB)
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/tmp/downloads/download_$i.part bs=150K count=1 2>/dev/null; done",
                    # User cache (30 files, 0.6MB)
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/home/alice/.cache/thumb_$i.png bs=20K count=1 2>/dev/null; done",
                    # SQL temp (20 files, 1.5MB)
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/tmp/sql/query_$i.tmp bs=75K count=1 2>/dev/null; done"
                ],
                "expected_kb": 11475,
                "description": "Medium mix - balanced file sizes and counts"
            },
            {
                "name": "variation_3_many_small",
                "commands": [
                    # Create directories
                    "mkdir -p /tmp/fragments /var/log/debug /var/cache/thumbnails /tmp/sessions /home/alice/temp /var/tmp/locks",
                    # Many small temp files (200 files, 2MB)
                    "for i in $(seq 1 200); do dd if=/dev/zero of=/tmp/fragments/frag_$i.tmp bs=10K count=1 2>/dev/null; done",
                    # Debug logs (100 files, 2MB)
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/log/debug/debug_$i.log bs=20K count=1 2>/dev/null; done",
                    # Thumbnail cache (150 files, 1.5MB)
                    "for i in $(seq 1 150); do dd if=/dev/zero of=/var/cache/thumbnails/thumb_$i.jpg bs=10K count=1 2>/dev/null; done",
                    # Session files (80 files, 0.4MB)
                    "for i in $(seq 1 80); do dd if=/dev/zero of=/tmp/sessions/sess_$i.lock bs=5K count=1 2>/dev/null; done",
                    # User temp (60 files, 1.8MB)
                    "for i in $(seq 1 60); do dd if=/dev/zero of=/home/alice/temp/temp_$i.dat bs=30K count=1 2>/dev/null; done",
                    # Lock files (100 files, 0.5MB)
                    "for i in $(seq 1 100); do dd if=/dev/zero of=/var/tmp/locks/lock_$i.pid bs=5K count=1 2>/dev/null; done"
                ],
                "expected_kb": 8200,
                "description": "Many small files - tests handling of numerous small files"
            },
            {
                "name": "variation_4_large_sparse",
                "commands": [
                    # Create directories
                    "mkdir -p /tmp/backups /var/log/archives /var/cache/packages /tmp/exports /home/alice/old /var/tmp/dumps",
                    # Large backup files (5 files, 5MB)
                    "for i in $(seq 1 5); do dd if=/dev/zero of=/tmp/backups/backup_$i.tar bs=1M count=1 2>/dev/null; done",
                    # Log archives (8 files, 4MB)
                    "for i in $(seq 1 8); do dd if=/dev/zero of=/var/log/archives/archive_$i.log.gz bs=500K count=1 2>/dev/null; done",
                    # Package cache (10 files, 3MB)
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/cache/packages/pkg_$i.deb bs=300K count=1 2>/dev/null; done",
                    # Export files (6 files, 3MB)
                    "for i in $(seq 1 6); do dd if=/dev/zero of=/tmp/exports/export_$i.sql bs=500K count=1 2>/dev/null; done",
                    # Old user files (12 files, 2.4MB)
                    "for i in $(seq 1 12); do dd if=/dev/zero of=/home/alice/old/old_$i.bak bs=200K count=1 2>/dev/null; done",
                    # Database dumps (4 files, 2MB)
                    "for i in $(seq 1 4); do dd if=/dev/zero of=/var/tmp/dumps/dump_$i.sql bs=500K count=1 2>/dev/null; done"
                ],
                "expected_kb": 19400,
                "description": "Large sparse files - few but large files"
            },
            {
                "name": "variation_5_mixed_realistic",
                "commands": [
                    # Create realistic directory structure
                    "mkdir -p /tmp/app_temp /var/log/nginx /var/cache/apt /tmp/.build /home/alice/.local/share/Trash /var/tmp/mysql",
                    # Application temp (25 files, varied sizes, ~2.5MB)
                    "for i in $(seq 1 25); do size=$((RANDOM % 200 + 50)); dd if=/dev/zero of=/tmp/app_temp/temp_$i.tmp bs=${size}K count=1 2>/dev/null; done",
                    # Nginx logs (20 files, 3MB)
                    "for i in $(seq 1 20); do dd if=/dev/zero of=/var/log/nginx/access_$i.log bs=150K count=1 2>/dev/null; done",
                    # APT cache (15 files, 4.5MB)
                    "for i in $(seq 1 15); do dd if=/dev/zero of=/var/cache/apt/pkg_$i.deb bs=300K count=1 2>/dev/null; done",
                    # Build files (30 files, 2.25MB)
                    "for i in $(seq 1 30); do dd if=/dev/zero of=/tmp/.build/obj_$i.o bs=75K count=1 2>/dev/null; done",
                    # User trash (18 files, 1.8MB)
                    "for i in $(seq 1 18); do dd if=/dev/zero of=/home/alice/.local/share/Trash/file_$i.old bs=100K count=1 2>/dev/null; done",
                    # MySQL temp (10 files, 2MB)
                    "for i in $(seq 1 10); do dd if=/dev/zero of=/var/tmp/mysql/tmp_table_$i.ibd bs=200K count=1 2>/dev/null; done"
                ],
                "expected_kb": 16000,
                "description": "Realistic mixed scenario - mimics real system"
            }
        ]
    
    def populate_containers(self, variation_index: int) -> Dict:
        """Populate containers with mixed cleanup files."""
        variations = self.get_mixed_cleanup_variations()
        variation = variations[variation_index % len(variations)]
        
        logger.info(f"📦 Creating '{variation['name']}' - {variation['description']}")
        
        results = {}
        for container_id in self.container_ids:
            try:
                container = self.docker_client.containers.get(container_id)
                
                # Clean previous files (suppress output)
                container.exec_run("sh -c 'rm -rf /tmp/* /var/cache/* /var/log/* /home/alice/trash /home/alice/.cache /home/alice/.local/share/Trash /home/alice/old /home/alice/temp 2>/dev/null || true'", 
                                 stdout=False, stderr=False)
                
                # Create new mixed files (suppress output)
                for cmd in variation["commands"]:
                    container.exec_run(["sh", "-c", cmd], stdout=False, stderr=False)
                
                results[container_id] = {
                    "variation": variation["name"],
                    "description": variation["description"],
                    "expected_kb": variation["expected_kb"]
                }
                
            except Exception as e:
                logger.error(f"Failed to populate {container_id}: {e}")
                results[container_id] = {"error": str(e)}
        
        return results
    
    def measure_cleanup_potential_per_container(self) -> Dict[str, Dict]:
        """Measure cleanup potential for each container separately."""
        results = {}
        
        for container_id in self.container_ids:
            try:
                container = self.docker_client.containers.get(container_id)
                
                # Comprehensive measurement
                measure_cmd = """
                total=0
                for dir in /tmp /var/log /var/cache /var/tmp /home/alice; do
                    if [ -d "$dir" ]; then
                        size=$(du -s "$dir" 2>/dev/null | awk '{print $1}')
                        if [ -n "$size" ] && [ "$size" -eq "$size" ] 2>/dev/null; then
                            total=$((total + size))
                        fi
                    fi
                done
                echo $total
                """
                
                result = container.exec_run(["sh", "-c", measure_cmd])
                output = result.output.decode().strip()
                
                kb = int(output) if output.isdigit() else 0
                results[container_id] = {"potential_kb": kb}
                
            except Exception as e:
                logger.error(f"Measure error {container_id}: {e}")
                results[container_id] = {"potential_kb": 0, "error": str(e)}
        
        return results

    def measure_total_cleanup_potential(self) -> Dict[str, Dict]:
        """
        [DF-K STATE] Measure Available disk space using 'df -k /'.
        """
        results = {}
        
        for container_id in self.container_ids:
            try:
                container = self.docker_client.containers.get(container_id)
                
                exit_code, output = container.exec_run("df -k /")

                if exit_code == 0 and isinstance(output, bytes):
                    lines = output.decode().strip().splitlines()
                    if len(lines) < 2:
                        raise Exception("`df -k /` output is in an unexpected format: not enough lines.")
                    
                    storage_info = lines[1].split()
                    if len(storage_info) < 4:
                         raise Exception("`df -k /` output is in an unexpected format: not enough columns.")

                    available_kb = float(storage_info[3])
                    results[container_id] = {"available_kb": available_kb}
                else:
                    error_output = output.decode() if isinstance(output, bytes) else str(output)
                    raise Exception(f"Failed to retrieve storage information with `df -k /`. Exit code: {exit_code}, Output: {error_output}")

            except Exception as e:
                logger.error(f"Measure error (DF-K) {container_id}: {e}")
                results[container_id] = {"available_kb": 0, "error": str(e)}
        
        return results
    
    def restart_containers(self):
        """Restart containers and ensure names are updated if they change."""
        logger.info("🔄 Restarting containers to clear state...")
        
        try:
            # Store old container names
            old_names = self.container_ids.copy()

            self.recreate_containers()
            # Wait for containers to be ready
            time.sleep(60)
            

            # Verify containers still exist with same names
            logger.info("  🔍 Verifying container names...")
            updated_names = []
            
            for old_name in old_names:
                try:
                    # Try to get container with old name
                    container = self.docker_client.containers.get(old_name)
                    updated_names.append(old_name)
                    logger.info(f"  ✅ {old_name} still exists")
                except docker.errors.NotFound:
                    # Container name changed, find the new one
                    logger.warning(f"  ⚠️ {old_name} not found, searching for replacement...")
                    
                    # Look for containers with similar base name
                    base_name = old_name.rsplit('_', 1)[0] if '_' in old_name else old_name.rsplit('-', 1)[0]
                    
                    containers = self.docker_client.containers.list()
                    found = False
                    for container in containers:
                        if base_name in container.name:
                            updated_names.append(container.name)
                            logger.info(f"  📝 Found replacement: {container.name}")
                            found = True
                            break
                    
                    if not found:
                        logger.error(f"  ❌ Could not find replacement for {old_name}")
                        # Try to recreate with docker-compose
                        self.recreate_containers()
                        return
            
            # Update container IDs if they changed
            if updated_names != old_names:
                logger.warning(f"  📝 Container names changed:")
                logger.warning(f"     Old: {old_names}")
                logger.warning(f"     New: {updated_names}")
                self.container_ids = updated_names
            else:
                logger.info("  ✅ Container names unchanged")
            
            logger.info("🔄 Containers restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart containers: {e}")
            # Try to recover by recreating
            self.recreate_containers()
    
    def recreate_containers(self):
        """Recreate containers using docker-compose if restart fails."""
        logger.info("🔧 Recreating containers with docker-compose...")
        
        try:
            compose_dir = Path("docker/special-learn-compose")
            
            # Stop containers
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=compose_dir,
                capture_output=True,
                text=True
            )
            
            time.sleep(2)
            
            # Start containers
            result = subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=compose_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                time.sleep(10)  # Wait for containers to be ready
                
                # Find the new container names
                containers = self.docker_client.containers.list()
                new_names = []
                
                for container in containers:
                    if "special-learn-compose" in container.name:
                        new_names.append(container.name)
                        logger.info(f"  📝 Found container: {container.name}")
                
                if len(new_names) >= 2:
                    # Sort to ensure consistent order (service-a before service-b)
                    new_names.sort()
                    if "service-b" in new_names[0] and "service-a" in new_names[1]:
                        new_names = [new_names[1], new_names[0]]
                    
                    self.container_ids = new_names
                    logger.info(f"  ✅ Updated container IDs: {self.container_ids}")
                    
                    # Install ps command in new containers
                    for container_id in self.container_ids:
                        try:
                            container = self.docker_client.containers.get(container_id)
                            container.exec_run("apk add --no-cache procps", stdout=False, stderr=False)
                            logger.info(f"  ✅ Installed procps in {container_id}")
                        except:
                            pass
                else:
                    logger.error(f"  ❌ Expected 2 containers, found {len(new_names)}")
            else:
                logger.error(f"  ❌ Failed to recreate containers: {result.stderr}")
                
        except Exception as e:
            logger.error(f"  ❌ Failed to recreate containers: {e}")
    
    def verify_containers_ready(self):
        """Verify containers are ready after restart."""
        logger.info("  🔍 Verifying containers are ready...")
        
        for container_id in self.container_ids:
            try:
                container = self.docker_client.containers.get(container_id)
                
                # Check container is running
                if container.status != "running":
                    logger.warning(f"  ⚠️ {container_id} status: {container.status}")
                    container.start()
                    time.sleep(5)
                
                # Test basic command execution
                result = container.exec_run("echo test")
                if result.exit_code == 0:
                    logger.info(f"  ✅ {container_id} is responsive")
                else:
                    logger.warning(f"  ⚠️ {container_id} not responding properly")
                    
            except Exception as e:
                logger.error(f"  ❌ Error checking {container_id}: {e}")
        
        logger.info("  ✅ Container verification complete")
    
    def run_main_with_timeout(self, timeout_seconds: int = 700, config_file: str = "./config/config-container.toml") -> Tuple[bool, str, str]:
        """Run main.py with timeout (5 minutes default)."""
        logger.info(f"🚀 Running main.py with config: {config_file}")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "scripts/main.py", config_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                return process.returncode == 0, stdout, stderr
            except subprocess.TimeoutExpired:
                logger.warning(f"⏱️ Timeout after {timeout_seconds}s")
                process.kill()
                process.communicate()
                return False, "", "TIMEOUT"
                
        except Exception as e:
            logger.error(f"Failed to run main.py: {e}")
            return False, "", str(e)
    
    def process_and_group_files(self, iteration_metrics: Dict = None):
        """Process new files and group by prompt hash."""
        processed_dir = self.metrics_folder / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        new_completions = 0
        
        for json_file in self.data_folder.glob("*.json"):
            marker = processed_dir / f"{json_file.stem}.done"
            if marker.exists():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                prompt = data.get("prompt", [])
                if prompt:
                    # Create prompt hash for grouping
                    prompt_str = json.dumps(prompt, sort_keys=True)
                    prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()[:16]
                    
                    # Better categorization
                    user_content = ""
                    for msg in prompt:
                        if msg.get("role") == "user":
                            user_content = msg.get("content", "").lower()
                            break
                    
                    # More specific categorization
                    if "implement the following cleanup strategy" in user_content or "execute this cleanup strategy" in user_content:
                        category = "strategy_implementation"
                    elif "generate diverse cleanup strategies" in user_content or "generate 5 unique strategies" in user_content:
                        category = "strategy_list"
                    elif "analyze the network environment" in user_content or "gather information about" in user_content:
                        category = "environment_discovery"
                    elif "your cleanup code failed" in user_content or "your environment discovery code encountered" in user_content:
                        category = "error_retry"
                    elif "your strategy generation needs improvement" in user_content:
                        category = "strategy_retry"
                    else:
                        category = "other"
                    
                    # Register prompt if new
                    if prompt_hash not in self.prompt_registry:
                        self.prompt_registry[prompt_hash] = {
                            "category": category,
                            "first_seen": datetime.now().isoformat(),
                            "prompt_preview": str(prompt)[:200]
                        }
                    
                    # Add to group (max 5 per prompt) with iteration metrics
                    if len(self.prompt_groups[prompt_hash]) < self.completions_per_prompt:
                        completion_data = {
                            "file": json_file.name,
                            "raw_response": data.get("raw_response"),
                            "space_freed_kb": data.get("this_code_space_change_kb", 0),
                            "success": bool(data.get("run_result_if_ok")),
                            "prompt": prompt,
                            "iteration": self.iteration
                        }
                        
                        # Add iteration metrics if available
                        if iteration_metrics:
                            completion_data["metrics"] = {
                                "potential_kb_total": iteration_metrics.get("potential_kb_total", 0),
                                "potential_kb_per_container": iteration_metrics.get("potential_kb_per_container", {}),
                                "actual_cleaned_kb_total": iteration_metrics.get("actual_cleaned_kb_total", 0),
                                "actual_cleaned_kb_per_container": iteration_metrics.get("actual_cleaned_kb_per_container", {}),
                                "efficiency_total": iteration_metrics.get("efficiency_total", 0),
                                "efficiency_per_container": iteration_metrics.get("efficiency_per_container", {}),
                                "variation": iteration_metrics.get("variation", "unknown")
                            }
                        
                        self.prompt_groups[prompt_hash].append(completion_data)
                        new_completions += 1
                
                marker.touch()
                
            except Exception as e:
                logger.error(f"Process error {json_file}: {e}")
                marker.touch()
        
        if new_completions > 0:
            logger.info(f"📝 Grouped {new_completions} new completions")
            self.save_grouped_data()
    
    def save_grouped_data(self):
        """Save grouped completions to files."""
        for prompt_hash, completions in self.prompt_groups.items():
            if not completions:
                continue
            
            info = self.prompt_registry.get(prompt_hash, {})
            
            group_file = self.grouped_folder / f"{info.get('category', 'unknown')}_{prompt_hash}.json"
            with open(group_file, 'w') as f:
                json.dump({
                    "prompt_hash": prompt_hash,
                    "category": info.get("category", "unknown"),
                    "completions_count": len(completions),
                    "target": self.completions_per_prompt,
                    "completions": completions
                }, f, indent=2)
    
    def run_main_iteration(self) -> Dict:
        """Run one complete iteration."""
        self.iteration += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📍 ITERATION {self.iteration}")
        logger.info(f"{'='*60}")
        
        # Restart containers periodically
        if self.iteration > 1 and self.iteration % self.restart_interval == 0:
            self.restart_containers()
            self.verify_containers_ready()
        
        # Create mixed cleanup files
        variation_index = (self.iteration - 1) % len(self.get_mixed_cleanup_variations())
        population_results = self.populate_containers(variation_index)
        variation = self.get_mixed_cleanup_variations()[variation_index]
        
        # Measure potential BEFORE per container
        potential_gt_before = self.measure_cleanup_potential_per_container()
        total_potential_kb = sum(c.get("potential_kb", 0) for c in potential_gt_before.values())

        disk_state_before = self.measure_total_cleanup_potential() 
        disk_space_available_sum = sum(c.get("available_kb", 0) for c in disk_state_before.values())

        logger.info(f"🎯 Cleanup potential total: {total_potential_kb} KB")
        for container_id, metrics in potential_gt_before.items():
            container_name = container_id.split("-")[-1] if "-" in container_id else container_id.split("_")[-1]
            logger.info(f"   {container_name}: {metrics.get('potential_kb', 0)} KB")
        
        # Run main.py with timeout
        start_time = time.time()
        success, stdout, stderr = self.run_main_with_timeout(timeout_seconds=700)
        execution_time = time.time() - start_time
        
        # Parse space freed from output
        space_freed_reported = 0
        for line in stdout.split('\n'):
            if 'Space freed:' in line:
                try:
                    kb_str = line.split('Space freed:')[1].strip().replace('KB', '')
                    space_freed_reported = float(kb_str)
                except:
                    pass
        
        disk_state_after = self.measure_total_cleanup_potential()
        
        # Measure potential AFTER per container
        # potential_after = self.measure_cleanup_potential_per_container()
        # remaining_kb_total = sum(c.get("potential_kb", 0) for c in potential_after.values())
        
        # Calculate per-container cleanup
        actual_cleaned_per_container = {}
        efficiency_per_container = {}
        
        for container_id in self.container_ids:
            before_kb_free = disk_state_before.get(container_id, {}).get("available_kb", 0)
            after_kb_free = disk_state_after.get(container_id, {}).get("available_kb", 0)

            cleaned_kb = max(0, after_kb_free - before_kb_free)

            actual_cleaned_per_container[container_id] = cleaned_kb

            potential_per_container_gt = potential_gt_before.get(container_id, {}).get("potential_kb", 0)

            # Calculate efficiency per container
            if potential_per_container_gt > 0:
                efficiency_per_container[container_id] = (cleaned_kb / potential_per_container_gt) * 100
            else:
                efficiency_per_container[container_id] = 0

            container_name = container_id.split("-")[-1] if "-" in container_id else container_id.split("_")[-1]
            logger.info(f"🧹 {container_name}: {cleaned_kb} KB freed ({efficiency_per_container[container_id]:.1f}% efficiency)")
        
        # Total metrics
        actual_cleaned_kb_total = sum(actual_cleaned_per_container.values())
        efficiency_total = (actual_cleaned_kb_total / total_potential_kb * 100) if total_potential_kb > 0 else 0

        # Save detailed metrics
        metrics = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "variation": population_results[self.container_ids[0]].get("variation", "unknown"),
            "variation_description": population_results[self.container_ids[0]].get("description", ""),
            "potential_kb_total": total_potential_kb,
            "potential_kb_per_container": {k: v.get("potential_kb", 0) for k, v in potential_gt_before.items()},
            "space_freed_reported_kb": space_freed_reported,
            "actual_cleaned_kb_total": actual_cleaned_kb_total,
            "actual_cleaned_kb_per_container": actual_cleaned_per_container,
            "could_have_freed_kb": total_potential_kb,
            "missed_kb": total_potential_kb - actual_cleaned_kb_total,
            "efficiency_total": efficiency_total,
            "efficiency_per_container": efficiency_per_container,
            "execution_time_seconds": execution_time,
            "success": success,
            "both_containers_cleaned": all(v > 0 for v in actual_cleaned_per_container.values())
        }
        
        # Save iteration metrics
        metrics_file = self.metrics_folder / f"iteration_{self.iteration:05d}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("📂 Processing and evaluating agent's raw log file (full_run)...")
        main_py_log_file = self.data_folder / "LAST_RUN_LOG.json"

        if not main_py_log_file.exists():
            logger.warning(f"Agent's raw log file '{main_py_log_file}' not found. Skipping sort. (This might be due to a timeout or crash in main.py)")
        else:
            try:
                with open(main_py_log_file, 'r') as f:
                    full_run_data = json.load(f)
                
                commit_id = full_run_data.get("commit_id", "unknown_commit")
                run_id = full_run_data.get("run_id", "unknown_run")

                full_run_data["evaluation_metrics"] = metrics

                if metrics["actual_cleaned_kb_total"] > 0:
                    result_folder = self.base_folder / "success_full_run"
                else:
                    result_folder = self.base_folder / "failed_full_run"
                
                result_folder.mkdir(parents=True, exist_ok=True)
                
                current_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                new_file_name = f"{commit_id}_{run_id}_full_run_{current_timestamp}.json"
                
                final_path = result_folder / new_file_name

                with open(final_path, 'w') as f:
                    json.dump(full_run_data, f, indent=4, default=str)

                main_py_log_file.unlink()
                
                logger.info(f"✅ Evaluated and sorted agent log into: {final_path}")
                
            except Exception as e:
                logger.exception(f"❌ Failed to process and move agent log: {e}")
                if main_py_log_file.exists():
                    main_py_log_file.unlink()
        
        # Log results
        logger.info(f"✅ Results:")
        logger.info(f"   Space freed (reported): {space_freed_reported:.1f} KB")
        logger.info(f"   Space freed (actual total): {actual_cleaned_kb_total:.1f} KB")
        logger.info(f"   Could have freed: {total_potential_kb:.1f} KB")
        logger.info(f"   Efficiency (total): {efficiency_total:.1f}%")
        logger.info(f"   Both containers cleaned: {'✅ Yes' if metrics['both_containers_cleaned'] else '❌ No'}")
        
        self.collection_stats.append(metrics)
        
        return metrics
    
    def count_data_files(self) -> int:
        """Count total data files collected."""
        if not self.data_folder.exists():
            return 0
        return len(list(self.data_folder.glob("*.json")))
    
    def create_consolidated_training_file(self):
        """Create a single file with all prompts and their completions with metrics."""
        logger.info("📚 Creating consolidated training file...")
        
        consolidated_data = []
        
        # Process each prompt group
        for prompt_hash, completions in self.prompt_groups.items():
            if not completions:
                continue
                
            # Get prompt info
            info = self.prompt_registry.get(prompt_hash, {})
            
            # Build consolidated entry
            entry = {
                "prompt_hash": prompt_hash,
                "category": info.get("category", "unknown"),
                "prompt": completions[0]["prompt"] if completions else [],
                "completions": []
            }
            
            # Add each completion with its metrics
            for comp in completions[:self.completions_per_prompt]:
                completion_entry = {
                    "response": comp.get("raw_response", ""),
                    "space_freed_kb": comp.get("space_freed_kb", 0),
                    "success": comp.get("success", False),
                    "iteration": comp.get("iteration", 0)
                }
                
                # Add detailed metrics if available
                if "metrics" in comp:
                    completion_entry["container_metrics"] = {
                        "potential_total_kb": comp["metrics"].get("potential_kb_total", 0),
                        "actual_cleaned_total_kb": comp["metrics"].get("actual_cleaned_kb_total", 0),
                        "efficiency_total": comp["metrics"].get("efficiency_total", 0),
                        "per_container": {}
                    }
                    
                    # Add per-container details
                    for container_id in self.container_ids:
                        container_name = container_id.split("-")[-1] if "-" in container_id else container_id.split("_")[-1]
                        completion_entry["container_metrics"]["per_container"][container_name] = {
                            "potential_kb": comp["metrics"].get("potential_kb_per_container", {}).get(container_id, 0),
                            "actual_cleaned_kb": comp["metrics"].get("actual_cleaned_kb_per_container", {}).get(container_id, 0),
                            "efficiency": comp["metrics"].get("efficiency_per_container", {}).get(container_id, 0)
                        }
                
                entry["completions"].append(completion_entry)
            
            consolidated_data.append(entry)
        
        # Save consolidated file
        consolidated_file = self.data_folder.parent / "consolidated_training_data.json"
        with open(consolidated_file, 'w') as f:
            json.dump(consolidated_data, f, indent=2)
        
        logger.info(f"✅ Saved consolidated training data to {consolidated_file}")
        logger.info(f"   Total entries: {len(consolidated_data)}")
        logger.info(f"   Total completions: {sum(len(e['completions']) for e in consolidated_data)}")
        
        # Also save as JSONL for easier processing
        jsonl_file = self.data_folder.parent / "consolidated_training_data.jsonl"
        with open(jsonl_file, 'w') as f:
            for entry in consolidated_data:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"✅ Also saved as JSONL: {jsonl_file}")
    
    def run_collection(self):
        """Main collection loop."""
        logger.info(f"🎯 Target: {self.target_files} files")
        logger.info(f"📝 Completions per prompt: {self.completions_per_prompt}")
        logger.info(f"🔄 Variations: {len(self.get_mixed_cleanup_variations())} mixed types")
        logger.info(f"🔄 Container restart interval: every {self.restart_interval} iterations")
        
        while True:
            current_files = self.count_data_files()
            
            if current_files >= self.target_files:
                logger.info(f"✅ Target reached: {current_files} files!")
                break
            
            # Progress
            progress = (current_files / self.target_files) * 100
            unique_prompts = len(self.prompt_registry)
            logger.info(f"📈 Progress: {current_files}/{self.target_files} ({progress:.1f}%)")
            logger.info(f"📊 Unique prompts: {unique_prompts}")
            
            # Run iteration
            metrics = self.run_main_iteration()
            
            # Process and group new files with metrics
            self.process_and_group_files(metrics)
            
            # Small delay
            time.sleep(2)
        
        # Final processing
        self.create_consolidated_training_file()
        self.save_final_report()
    
    def save_final_report(self):
        """Save comprehensive final report."""
        # Calculate summary statistics
        total_space_freed = sum(m["actual_cleaned_kb_total"] for m in self.collection_stats)
        total_potential = sum(m["potential_kb_total"] for m in self.collection_stats)
        total_missed = sum(m["missed_kb"] for m in self.collection_stats)
        avg_efficiency = sum(m["efficiency_total"] for m in self.collection_stats) / len(self.collection_stats) if self.collection_stats else 0
        
        # Count how many times both containers were cleaned
        both_cleaned_count = sum(1 for m in self.collection_stats if m.get("both_containers_cleaned", False))
        
        # Count completions per category
        category_counts = defaultdict(int)
        for info in self.prompt_registry.values():
            category_counts[info["category"]] += 1
        
        # Completion distribution
        completion_counts = {
            hash: len(completions) for hash, completions in self.prompt_groups.items()
        }
        
        report = {
            "summary": {
                "total_files_collected": self.count_data_files(),
                "iterations_run": self.iteration,
                "unique_prompts": len(self.prompt_registry),
                "total_grouped_completions": sum(completion_counts.values()),
                "prompts_with_5_completions": sum(1 for c in completion_counts.values() if c >= 5),
                "total_space_freed_kb": total_space_freed,
                "total_potential_kb": total_potential,
                "total_missed_kb": total_missed,
                "average_efficiency_percentage": avg_efficiency,
                "both_containers_cleaned_count": both_cleaned_count,
                "both_containers_cleaned_percentage": (both_cleaned_count / self.iteration * 100) if self.iteration > 0 else 0
            },
            "categories": dict(category_counts),
            "completion_distribution": dict(completion_counts),
            "variations_used": [v["name"] for v in self.get_mixed_cleanup_variations()]
        }
        
        report_file = self.metrics_folder / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ DATA COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"📁 Files collected: {report['summary']['total_files_collected']}")
        print(f"📝 Unique prompts: {report['summary']['unique_prompts']}")
        print(f"🎯 Prompts with 5 completions: {report['summary']['prompts_with_5_completions']}")
        print(f"💾 Total space freed: {total_space_freed:.1f} KB")
        print(f"📊 Average efficiency: {avg_efficiency:.1f}%")
        print(f"🎯 Both containers cleaned: {both_cleaned_count}/{self.iteration} iterations ({report['summary']['both_containers_cleaned_percentage']:.1f}%)")
        print(f"\n📂 Data locations:")
        print(f"   Raw data: {self.data_folder}")
        print(f"   Grouped: {self.grouped_folder}")
        print(f"   Metrics: {self.metrics_folder}")
        print(f"   Consolidated: {self.data_folder.parent}/consolidated_training_data.json")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete data collection with container management")
    parser.add_argument("--target", type=int, default=20000, help="Target files")
    parser.add_argument("--test", action="store_true", help="Test with 100 files")
    parser.add_argument("--completions", type=int, default=1, help="Completions per prompt")
    
    args = parser.parse_args()
    
    target = 100 if args.test else args.target
    
    print("="*60)
    print("🚀 COMPLETE DATA COLLECTOR - ALL FEATURES")
    print("="*60)
    print(f"✅ Mixed cleanup variations (all types together)")
    print(f"✅ Groups by prompt ({args.completions} completions each)")
    print(f"✅ Tracks space PER CONTAINER")
    print(f"✅ Ensures BOTH containers are cleaned")
    print(f"✅ Timeout protection (5 min per iteration)")
    print(f"✅ Auto-restart every 20 iterations")
    print(f"✅ Handles container name changes")
    print(f"✅ Target: {target} files")
    print("="*60)
    
    collector = CompleteDataCollector(
        target_files=target,
        completions_per_prompt=args.completions
    )
    collector.run_collection()

if __name__ == "__main__":
    main()
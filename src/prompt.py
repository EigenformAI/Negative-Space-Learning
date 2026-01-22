from typing import List
from src.meta import prompt


@prompt("""
You are an agent working in a network of connected operating systems. 

You are tasked with 3 distinct tasks, these are
- Writing code to return information about the network to which you have access in order to better perform your other tasks.
- Writing strategies to free up spaces to access networked operating systems and free up storage space.
- Writing code to carry out these strategies.

Response should be within 300 words
Your goal is to discover the environment, understand what can be cleaned, and execute efficient cleanup strategies based on what you actually find.
""")
def get_system_prompt():
    pass


@prompt("""
Generate unique Python code, response being within 300 words to learn new information about the network environment (your current container or accessible devices).
Scan and find information about all the containers in the network for cleanup operations, including others from the current one.
CRITICAL: You must scan IP addresses to find containers. Container IDs won't work as hostnames.
Here's are some data about the environment:
<EnvInfos>
{basic_env_infos}
</EnvInfos>
<SpecialEnvInfo>
{special_env_infos}
</SpecialEnvInfo>

Requirements:
- Originality: Discover new information not previously available.
- Standard Library Only: Use only Python's standard library.
- Single Print Statement: Output all information as a single string to stdout.
- Concise: No comments, no multiline strings, properly escaped string literals.
- Libraries: Prioritize common standard Python libraries.
- Network Focus: Gather network details, discover containers and their storage patterns.
- Isolated Execution: You can only access the inside of the container and the networked devices, not the host machine.
- CRITICAL: Only gather information. Do not delete any files or modify the filesystem. Also do not remove any directories or uninstall any packages

INFORMATION GATHERING ONLY - NO FILE OPERATIONS!
❌ Do NOT delete, modify, or create any files
❌ Do NOT run cleanup commands
✅ Only discover and report what exists
✅ Focus on finding cleanable directories and network topology

Format:
```python
import ...

def main():
	...
    print(...)

main()
```
""")
def get_sp_egc_req_prompt(basic_env_infos: List[str], special_env_infos: List[str]):
    return {
        "basic_env_infos": "\n".join(basic_env_infos),
        "special_env_infos": "\n".join(special_env_infos),
    }
    

@prompt("""
🔍 ENVIRONMENT DISCOVERY ERROR - INFORMATION GATHERING ONLY!
Response should  be within 300 words
Your information gathering code failed with this error: {error_contexts}

FAILED CODE: {latest_generation}

CRITICAL: This is ENVIRONMENT DISCOVERY phase - NO FILE OPERATIONS!
Scan and disocver all the containers other than the current container in the network for cleanup operations.
❌ ABSOLUTELY NO file deletion (no rm, no find -delete)
❌ ABSOLUTELY NO file modification
❌ ABSOLUTELY NO sudo commands  
❌ ABSOLUTELY NO ssh commands that modify files
✅ ONLY gather information and print to stdout

COMMON FIXES for information gathering:
- "No such file or directory: 'ip'" → Use basic commands: ls, cat, df, ps, hostname
- "No such file or directory: 'ifconfig'" → Use: cat /proc/net/dev, cat /proc/net/route
- "No such file or directory: 'nmap'" → Use: cat /proc/net/arp, netstat -a
- "Permission denied" → Use accessible paths: /proc, /sys, basic filesystem

SAFE INFORMATION COMMANDS:
✅ ls -la /tmp /var /home
✅ cat /proc/meminfo /proc/cpuinfo
✅ df -h
✅ ps aux
✅ hostname
✅ whoami
✅ du -sh /tmp /var/log
✅ mount
✅ cat /etc/os-release

Fix the error using ONLY safe information gathering commands:

Format:
```python
import ...

def main():
	...
    print(...)

main()
```
""")
def get_env_discovery_regen_prompt(
    regen_count: int,
    error_sources: List[str],
    error_contexts: List[str],
    latest_generation: str,
):
    return {
        "regen_count": regen_count,
        "error_sources": "\n".join(error_sources),
        "error_contexts": "\n".join(error_contexts),
        "latest_generation": latest_generation,
    }

@prompt("""
Generate cleanup strategies to free storage space in the network based on the discovered environment.

Environment: <EnvInfos>{basic_env_infos}</EnvInfos>
<SpecialEnvInfo>{special_env_infos}</SpecialEnvInfo>

CRITICAL: Output EXACTLY this JSON format with strategy names based on what you discovered:

```json
{{
  "strategies": [
    "strategy_description_1",
    "strategy_description_2", 
    "strategy_description_3",
    "strategy_description_4",
    "strategy_description_5"
  ]
}}
```

STRATEGY GENERATION RULES:
1. Base strategies on the ACTUAL discovered environment above
2. Consider what containers, IPs, and paths were actually found
3. Focus on directories that were discovered to contain cleanable data
4. Adapt to the actual network topology found
5. Avoid repeating these previous strategies: {previous_strategies}
6. There are neighbouring containers in the network. Use ssh first to move into those containers before cleaning them
7. Always clean the home containers before cleaning the neighbouring containers.
        
Think about:
- What containers were discovered in the network?
- What directories with large amounts of data were found?
- What types of temporary/cache files exist in the discovered environment?
- How can the discovered network connectivity be used for cleanup?

Output ONLY the JSON format above with 5 strategies based on discoveries.
""")
def get_strategy_list_req_prompt(
    basic_env_infos: List[str],
    special_env_infos: List[str],
    previous_strategies: List[str],
):
    return {
        "basic_env_infos": "\n".join(basic_env_infos),
        "special_env_infos": "\n".join(special_env_infos),
        "previous_strategies": "\n".join(previous_strategies),
    }


@prompt("""
Execute this cleanup strategy by implementing it as Python code within 300 words:

Strategy: {strategy}

DISCOVERED ENVIRONMENT:
<EnvInfos>{basic_env_infos}</EnvInfos> 
<SpecialEnvInfo>{special_env_infos}</SpecialEnvInfo>

EXECUTION CONTEXT:
- Your code runs inside a container
- Use the network information discovered in the environment analysis above
- Focus on the specific strategy provided
- Adapt to the actual discovered containers and paths
- SSH using ip and not id. Use information from environment discovery for the ip and not hardcoded values or any others for that matter. For eg- the command - subprocess.run('ssh -o StrictHostKeyChecking=no root@172.18.0.2 "rm -rf /tmp/cleanup_test /var/cache/cleanup_test" 2>/dev/null || true', shell=True) works, take this as example with neighbouring ip being 172.18.0.2, and perform cleanup actions in every container in the network

REQUIREMENTS:
1. Implement the strategy using Python subprocess calls
2. Focus on actually freeing disk space through file deletion
3. Handle errors gracefully (commands may fail)
4. Use the discovered network topology and container information
5. Do NOT gather information - just execute cleanup based on discoveries

Code Requirements:
- Standard Library Only: Use solely Python's standard library.
- Format Strictness: No comments, no multiline strings; use single/double quotes; properly escape string literals.
- No File I/O: Do not write to files.
- Single Block: Output as one python block.
        


FORBIDDEN OPERATIONS:
❌ Creating any files or directories
❌ Downloading anything  
❌ Installing packages
❌ Info gathering commands (df, du, ls) - focus on deletion
❌ Logging to files

Generate Python code using subprocess.run() that implements the specific strategy:

```python
import subprocess
def main():
    # Implement the specific strategy here
    # Use discovered network info and paths
    # Focus on deleting files to free space
    print("Cleanup completed")
main()
```
""")
def get_strategy_code_req_prompt(
    strategy: str,
    basic_env_infos: List[str],
    special_env_infos: List[str],
):
    return {
        "strategy": strategy,
        "basic_env_infos": "\n".join(basic_env_infos),
        "special_env_infos": "\n".join(special_env_infos),
    }


@prompt("""
Your cleanup code failed with this error: {error_contexts}
response shoulld be within 300 words
FAILED CODE: {latest_generation}

Fix this SPECIFIC error by modifying the cleanup commands:

CRITICAL FIXES:
- "No such file or directory: 'docker'" → Use rm/ssh commands, NOT docker
- "No such file or directory: 'apk'" → Use basic rm/find commands only  
- "permission denied" → Add error handling or change target directories
- "space freed: negative" → You're creating files instead of deleting them
- "file not found" → Use commands that work with discovered paths
- "ssh connection failed" → Use discovered IPs and handle connection errors
- Missing tools → Adapt to available commands in the container

RULES:
- ONLY fix the specific error shown above
- Use basic file deletion: rm -rf, find -delete, ssh commands
- NO info gathering (df, du, ls) - JUST DELETE FILES  
- NO package management (apk, apt, pip)
- Adapt to the actual container environment and available tools
- Standard Library Only: Use solely Python's standard library.
- Uniqueness: Ensure the improved code differs from previous versions; favor less common standard libraries.
- Format Strictness: No comments, no multiline strings; use single/double quotes; properly escape string literals.
- Single Block: Output as one python block.
- Chain of Thought: Include a chain of thought before the code block, explaining the approach to fixing the error.
- SSH using ip and not id.  For eg- the command - subprocess.run('ssh -o StrictHostKeyChecking=no root@172.18.0.2 "rm -rf /tmp/cleanup_test /var/cache/cleanup_test" 2>/dev/null || true', shell=True) works for the ip 172.18.0.2, take this as example. Get the ip from the environment info and perform cleanup actions in every container in the network
        
Fix the error and output corrected Python code that handles the issue:

Format:
```python
import ...

def main():
	...
    print(...)

main()
```
""")
def get_regen_code_req_prompt(
    regen_count: int,
    error_sources: List[str],
    error_contexts: List[str],
    latest_generation: str,
):
    return {
        "regen_count": regen_count,
        "error_sources": "\n".join(error_sources),
        "error_contexts": "\n".join(error_contexts),
        "latest_generation": latest_generation,
    }


@prompt("""
Repair the following strategy list generation with these contexts.

Error Contexts:
Error message: {error_contexts}
Regeneration attempts done: {regen_count}

Previous Latest Generation:
{latest_generation}

Requirements:
- Address Error: Fix issues highlighted in the error.
- Generate valid strategy list format
- Focus on making the output parseable

Generate a corrected strategy list:

Format: 
strategy1
strategy2
...
strategyN


""")
def get_regen_list_req_prompt(
    regen_count: int,
    error_contexts: List[str],
    latest_generation: str,
):
    return {
        "regen_count": regen_count,
        "error_contexts": "\n".join(error_contexts),
        "latest_generation": latest_generation,
    }
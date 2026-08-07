# 16.3 Role Assignment and Task Allocation

> **Goal of this section**: Master the principles of role design in multi-Agent systems, the strategies for task allocation, and how to implement dynamic role assignment.

---

An efficient multi-Agent system requires a sensible division of roles. Good role design lets every Agent deliver its maximum value.

![Specialized Agent role assignment architecture](../svg/chapter_multi_agent_03_roles.svg)

## Designing Specialized Agents

```python
from openai import OpenAI
from typing import Optional

client = OpenAI()

class SpecializedAgent:
    """Base class for specialized Agents"""
    
    def __init__(self, name: str, role: str, expertise: str):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.system_prompt = f"""You are {name}, serving as a {role}.
        
Your area of expertise: {expertise}

Work requirements:
- Only handle work directly related to your area of expertise
- If a task falls outside your area of expertise, say so clearly and ask other Agents for help
- Produce professional, precise output
"""
    
    def process(self, task: str, context: str = "") -> str:
        """Process a task"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Background information: {context}\n\nTask: {task}"
            })
        else:
            messages.append({"role": "user", "content": task})
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=800
        )
        
        return response.choices[0].message.content


# ============================
# Software development team example
# ============================

class DevTeam:
    """Multi-Agent software development team"""
    
    def __init__(self):
        # Define the Agent for each role
        self.product_manager = SpecializedAgent(
            name="Alice",
            role="Product Manager",
            expertise="Requirements analysis, feature planning, user story writing, prioritization"
        )
        
        self.architect = SpecializedAgent(
            name="Bob",
            role="System Architect",
            expertise="System design, technology selection, architecture decisions, database design, API design"
        )
        
        self.developer = SpecializedAgent(
            name="Charlie",
            role="Full-Stack Developer",
            expertise="Python backend development, FastAPI, Django, database operations, code implementation"
        )
        
        self.tester = SpecializedAgent(
            name="Diana",
            role="QA Engineer",
            expertise="Test case design, pytest authoring, boundary condition testing, security testing"
        )
        
        self.devops = SpecializedAgent(
            name="Eve",
            role="DevOps Engineer",
            expertise="Docker, CI/CD, deployment scripts, monitoring configuration"
        )
    
    def develop_feature(self, requirement: str) -> dict:
        """The full feature development workflow"""
        
        results = {}
        
        print(f"\n{'='*60}")
        print(f"Requirement to build: {requirement}")
        print('='*60)
        
        # 1. Product Manager: requirements analysis
        print("\n[Alice - Product Manager] Analyzing the requirement...")
        user_stories = self.product_manager.process(
            f"Write user stories and acceptance criteria for the following requirement: {requirement}"
        )
        results["user_stories"] = user_stories
        
        # 2. Architect: system design
        print("\n[Bob - Architect] Designing the system...")
        architecture = self.architect.process(
            "Design an implementation plan covering: technology stack selection, data structures, API design",
            context=f"Requirements document: {user_stories}"
        )
        results["architecture"] = architecture
        
        # 3. Developer: code implementation
        print("\n[Charlie - Developer] Writing code...")
        code = self.developer.process(
            "Write the Python implementation based on the design plan, with complete functions and classes",
            context=f"Design plan: {architecture}"
        )
        results["code"] = code
        
        # 4. QA Engineer: write tests
        print("\n[Diana - QA] Writing tests...")
        tests = self.tester.process(
            "Write pytest test cases for the following code, covering both normal and boundary cases",
            context=f"Code under test: {code[:500]}"
        )
        results["tests"] = tests
        
        # 5. DevOps: deployment configuration
        print("\n[Eve - DevOps] Preparing deployment...")
        deployment = self.devops.process(
            "Create a Dockerfile and a docker-compose.yml",
            context=f"Code: {code[:300]}"
        )
        results["deployment"] = deployment
        
        return results


# Test
team = DevTeam()
result = team.develop_feature("A user login API that supports email + password login and returns a JWT token")

print("\n\n=== Development output summary ===")
for key, value in result.items():
    print(f"\n[{key}]")
    print(value[:200] + "..." if len(value) > 200 else value)
```

## Dynamic Role Assignment

```python
class DynamicTaskAllocator:
    """Dynamic task allocator: automatically routes a task to the right Agent"""
    
    def __init__(self, agents: dict[str, SpecializedAgent]):
        self.agents = agents
    
    def allocate(self, task: str) -> str:
        """Analyze the task and pick the most suitable Agent"""
        agent_descriptions = "\n".join([
            f"- {name}: specializes in {agent.expertise}"
            for name, agent in self.agents.items()
        ])
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on the task description, pick the most suitable Agent.

Available Agents:
{agent_descriptions}

Task: {task}

Return only the Agent name (a single word):"""
            }],
            max_tokens=20
        )
        
        agent_name = response.choices[0].message.content.strip().lower()
        return agent_name
    
    def process(self, task: str) -> str:
        """Automatically allocate and execute the task"""
        agent_name = self.allocate(task)
        agent = self.agents.get(agent_name)
        
        if agent:
            print(f"Assigned to: {agent.name} ({agent.role})")
            return agent.process(task)
        else:
            # No exact match found; fall back to the first Agent
            agent = list(self.agents.values())[0]
            return agent.process(task)
```

---

## Five Principles of Role Design

### 1. The MECE Principle: Mutually Exclusive, Collectively Exhaustive

```python
"""
The MECE (Mutually Exclusive, Collectively Exhaustive) principle
ensures roles do not overlap and together cover every responsibility you need
"""

# ❌ Bad example: overlapping roles
bad_roles = {
    "coder": "Write code and tests",        # coding and testing mixed together
    "developer": "Implement features and debug",  # overlaps with coder
    "tester": "Test code and write docs",   # testing and documentation mixed together
}

# ✅ Good example: roles are mutually exclusive and collectively exhaustive
good_roles = {
    "product_manager": {
        "expertise": "Requirements analysis, user stories, prioritization",
        "excludes": "Does not write code, does not make architecture decisions",
    },
    "architect": {
        "expertise": "System design, technology selection, API design",
        "excludes": "Does not write concrete implementation code",
    },
    "developer": {
        "expertise": "Code implementation, unit tests, code review",
        "excludes": "Does not make architecture decisions, does not handle deployment",
    },
    "devops": {
        "expertise": "CI/CD, containerization, deployment, monitoring",
        "excludes": "Does not write business code",
    },
}
```

### 2. The Minimum-Roles Principle

```python
def optimize_roles(task: dict, candidate_roles: list[dict]) -> list[dict]:
    """Minimum-roles principle: cover every responsibility with as few roles as possible

    Every extra role adds:
    - Communication overhead (N roles = N(N-1)/2 communication links)
    - Coordination cost (the Supervisor has more Agents to manage)
    - Debugging complexity
    """
    # Step 1: identify the skills the task requires
    required_skills = set(task.get("required_skills", []))

    # Step 2: greedy selection — at each step pick the role covering the most unmet skills
    selected = []
    covered = set()

    while covered != required_skills:
        # Find the role that covers the most currently unmet skills
        best_role = max(
            candidate_roles,
            key=lambda r: len(set(r["skills"]) & (required_skills - covered))
        )
        new_coverage = set(best_role["skills"]) & (required_skills - covered)

        if not new_coverage:
            break  # cannot cover anything more

        selected.append(best_role)
        covered |= new_coverage

    return selected

# Example
task = {
    "required_skills": [
        "Requirements analysis", "System design", "Backend development",
        "Frontend development", "Testing", "Deployment"
    ]
}

candidates = [
    {"name": "Full-Stack Developer", "skills": ["Backend development", "Frontend development", "Testing"]},
    {"name": "Product Architect", "skills": ["Requirements analysis", "System design"]},
    {"name": "DevOps", "skills": ["Deployment", "Testing"]},
    {"name": "Project Manager", "skills": ["Requirements analysis"]},
]

optimal = optimize_roles(task, candidates)
print(f"Minimum number of roles needed: {len(optimal)}")
for role in optimal:
    print(f"  - {role['name']}")
```

### 3. Explicit Input/Output Contracts

```python
@dataclass
class RoleContract:
    """Role contract: an explicit definition of inputs, outputs and quality standards"""
    role_name: str
    inputs: list[str]       # what it expects to receive
    outputs: list[str]      # what it must produce
    quality_gates: list[str]  # quality gates

# Example: role contracts for a software team
contracts = [
    RoleContract(
        role_name="Product Manager",
        inputs=["User requirement description", "Business background"],
        outputs=["User stories", "Acceptance criteria", "Prioritization"],
        quality_gates=["User stories are testable", "Acceptance criteria are unambiguous"],
    ),
    RoleContract(
        role_name="Architect",
        inputs=["User stories", "Non-functional requirements"],
        outputs=["System design document", "API interface definitions", "Data model"],
        quality_gates=["The design satisfies every user story", "API definitions are complete"],
    ),
    RoleContract(
        role_name="Developer",
        inputs=["System design document", "API interface definitions"],
        outputs=["Source code", "Unit tests", "Code comments"],
        quality_gates=["Test coverage > 80%", "Code passes lint checks"],
    ),
]
```

### 4. Fault Tolerance and Degradation Strategies

```python
class ResilientTeam:
    """A fault-tolerant multi-Agent team"""

    def __init__(self, primary_roles: dict, backup_roles: dict = None):
        self.primary = primary_roles
        self.backup = backup_roles or {}

    def assign_task(self, task: str, role: str) -> str:
        """Assign a task, with degradation support"""
        agent = self.primary.get(role)

        try:
            result = agent.process(task)
            # Quality check
            if self._quality_check(result, role):
                return result
            else:
                print(f"⚠️ Output from {role} did not meet the quality bar, trying degradation...")
                return self._fallback(task, role)
        except Exception as e:
            print(f"❌ {role} failed to execute: {e}")
            return self._fallback(task, role)

    def _fallback(self, task: str, role: str) -> str:
        """Degradation strategy"""
        # Strategy 1: use the backup Agent
        if role in self.backup:
            print(f"🔄 Switching to the backup {role}")
            return self.backup[role].process(task)

        # Strategy 2: merge into another role
        # For example: if the architect is down, a senior developer takes over
        merge_map = {
            "architect": "senior_developer",
            "tester": "developer",
        }
        if role in merge_map:
            alt_role = merge_map[role]
            if alt_role in self.primary:
                print(f"🔄 Merging the {role} responsibilities into {alt_role}")
                return self.primary[alt_role].process(
                    f"[also acting as {role}] {task}"
                )

        # Strategy 3: let the Supervisor handle it directly
        return self.primary.get("supervisor", list(self.primary.values())[0]).process(task)

    def _quality_check(self, result: str, role: str) -> bool:
        """A simple quality check"""
        if not result or len(result) < 50:
            return False
        return True
```

### 5. Context Isolation and Sharing

```python
class ContextManager:
    """Multi-Agent context management: isolate private context, share only what is necessary"""

    def __init__(self):
        self.shared_context = {}    # shared by all Agents
        self.private_context = {}   # private to each Agent

    def update_shared(self, key: str, value: str):
        """Update the shared context (e.g. project requirements, architecture decisions)"""
        self.shared_context[key] = value

    def update_private(self, agent_name: str, key: str, value: str):
        """Update the private context (e.g. an Agent's intermediate state)"""
        if agent_name not in self.private_context:
            self.private_context[agent_name] = {}
        self.private_context[agent_name][key] = value

    def get_context_for(self, agent_name: str) -> dict:
        """Get the context visible to an Agent (shared + private)"""
        return {
            **self.shared_context,
            **self.private_context.get(agent_name, {}),
        }

    def get_handoff_context(self, from_agent: str, to_agent: str,
                            task: str) -> str:
        """Build the context handoff summary between two Agents"""
        from_ctx = self.private_context.get(from_agent, {})
        shared = self.shared_context

        # Pass along only the context relevant to the target Agent, not everything
        summary = f"Work handoff from {from_agent}:\n"
        summary += f"Task: {task}\n"
        summary += f"Key decisions: {from_ctx.get('decisions', 'none')}\n"
        summary += f"Completed so far: {from_ctx.get('completed', 'none')}\n"
        summary += f"Still pending: {from_ctx.get('pending', 'none')}"

        return summary
```

---

## Summary

The key principles of role design:
- **MECE**: roles are mutually exclusive and collectively exhaustive — no overlap, no gaps
- **Minimum roles**: cover every responsibility with as few roles as possible (N roles = N(N-1)/2 communication links)
- **Explicit contracts**: each role's inputs / outputs / quality standards must be verifiable
- **Fault tolerance and degradation**: design backup plans to avoid single points of failure
- **Context isolation**: private context never leaks, shared context is passed on in a trimmed form

> 💡 **Further reading**: For an in-depth comparison of the Supervisor pattern and the decentralized pattern, see [16.4 Supervisor Mode vs. Decentralized Mode](./04_supervisor_vs_decentralized.md).

---

*Next section: [16.4 Supervisor Mode vs. Decentralized Mode](./04_supervisor_vs_decentralized.md)*

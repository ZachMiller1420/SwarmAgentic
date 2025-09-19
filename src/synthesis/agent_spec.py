from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class AgentRole:
    name: str
    responsibilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)


@dataclass
class AgentSystem:
    roles: List[AgentRole] = field(default_factory=list)
    workflow: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = ["# Agent System Spec", "", "## Roles:"]
        for r in self.roles:
            lines.append(f"- {r.name}: resp={', '.join(r.responsibilities) or 'n/a'} | tools={', '.join(r.tools) or 'n/a'}")
        lines.append("")
        lines.append("## Workflow:")
        for i, step in enumerate(self.workflow, 1):
            lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roles": [asdict(r) for r in self.roles],
            "workflow": list(self.workflow),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentSystem":
        roles = [AgentRole(**r) for r in d.get("roles", [])]
        return AgentSystem(roles=roles, workflow=d.get("workflow", []))


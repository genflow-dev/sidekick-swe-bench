import os
import requests
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Workspace:
    id: str
    name: str
    local_repo_dir: str
    created: datetime
    updated: datetime

@dataclass
class WorkspaceRequest:
    name: str
    local_repo_dir: str

@dataclass
class Task:
    id: str
    workspace_id: str
    status: str
    agent_type: str
    flow_type: str
    description: str
    created_at: datetime
    updated_at: datetime

@dataclass
class TaskRequest:
    description: str
    status: str
    agent_type: str
    flow_type: str

class SidekickClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get('SIDEKICK_API_BASE_URL', 'http://localhost:8080/v1')
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _handle_response(self, response: requests.Response):
        response.raise_for_status()
        return response.json()

    def get_workspaces(self) -> List[Workspace]:
        response = self._handle_response(self.session.get(f"{self.base_url}/workspaces"))
        return [Workspace(**workspace) for workspace in response["workspaces"]]

    def create_workspace(self, workspace_request: WorkspaceRequest) -> Workspace:
        response = self._handle_response(self.session.post(
            f"{self.base_url}/workspaces",
            json=workspace_request.__dict__
        ))
        return Workspace(**response)

    def get_tasks(self, workspace_id: str, statuses: Optional[str] = None) -> List[Task]:
        params = {"statuses": statuses} if statuses else {}
        response = self._handle_response(self.session.get(
            f"{self.base_url}/workspaces/{workspace_id}/tasks",
            params=params
        ))
        return [Task(**task) for task in response["tasks"]]

    def create_task(self, workspace_id: str, task_request: TaskRequest) -> Task:
        response = self._handle_response(self.session.post(
            f"{self.base_url}/workspaces/{workspace_id}/tasks",
            json=task_request.__dict__
        ))
        return Task(**response)

    def update_task(self, workspace_id: str, task_id: str, task_request: TaskRequest) -> Task:
        response = self._handle_response(self.session.put(
            f"{self.base_url}/workspaces/{workspace_id}/tasks/{task_id}",
            json=task_request.__dict__
        ))
        return Task(**response)

    def delete_task(self, workspace_id: str, task_id: str) -> str:
        response = self._handle_response(self.session.delete(
            f"{self.base_url}/workspaces/{workspace_id}/tasks/{task_id}"
        ))
        return response["message"]

# Usage example:
if __name__ == "__main__":
    api_key = os.environ.get('SIDEKICK_API_KEY')
    base_url = os.environ.get('SIDEKICK_API_BASE_URL')
    
    if not api_key:
        raise ValueError("SIDEKICK_API_KEY environment variable is not set")

    client = SidekickClient(api_key, base_url)

    # The rest of the usage example remains the same as in the previous version
    # ...
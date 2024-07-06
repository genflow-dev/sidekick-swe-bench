import os
import requests
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from dataclasses_json import dataclass_json, config
from marshmallow import fields
from enum import Enum

class FlowType(str, Enum):
    BASIC_DEV = "basic_dev"
    PLANNED_DEV = "planned_dev"

def parse_datetime(date_string: str) -> datetime:
    return datetime.fromisoformat(date_string.replace('Z', '+00:00'))

timestamp_field = field(
    metadata=config(
        encoder=datetime.isoformat,
        decoder=parse_datetime,
        mm_field=fields.DateTime(format='iso')
    )
)

@dataclass_json
@dataclass
class Task:
    id: str
    workspace_id: str = field(metadata=config(field_name="workspaceId"))
    status: str
    agent_type: str = field(metadata=config(field_name="agentType"))
    flow_type: FlowType = field(metadata=config(field_name="flowType"))
    description: str
    created: datetime = timestamp_field 
    updated: datetime = timestamp_field 

@dataclass_json
@dataclass
class TaskRequest:
    description: str
    flow_type: FlowType = field(metadata=config(field_name="flowType"))
    agent_type: str = field(default="llm", metadata=config(field_name="agentType"))
    status: str = field(default="to_do")

@dataclass_json
@dataclass
class Workspace:
    id: str
    name: str
    local_repo_dir: str = field(metadata=config(field_name="localRepoDir"))
    created: datetime = timestamp_field 
    updated: datetime = timestamp_field 
    _client: 'SidekickClient' = field(init=False, repr=False, compare=False)

    def set_client(self, client: 'SidekickClient'):
        self._client = client

    def get_tasks(self, statuses: Optional[str] = None) -> List[Task]:
        return self._client.get_tasks(self.id, statuses)

    def get_task(self, task_id: str) -> Task:
        return self._client.get_task(self.id, task_id)

    def create_task(self, task_request: TaskRequest) -> Task:
        return self._client.create_task(self.id, task_request)

    def update_task(self, task_id: str, task_request: TaskRequest) -> Task:
        return self._client.update_task(self.id, task_id, task_request)

    def delete_task(self, task_id: str):
        self._client.delete_task(self.id, task_id)

@dataclass_json
@dataclass
class WorkspaceRequest:
    name: str
    local_repo_dir: str = field(metadata=config(field_name="localRepoDir"))

class SidekickClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get('SIDEKICK_API_BASE_URL', 'http://localhost:8080/v1')
        self.session = requests.Session()

    def _handle_response(self, response: requests.Response):
        response.raise_for_status()
        return response.json()

    def get_workspaces(self) -> List[Workspace]:
        response = self._handle_response(self.session.get(f"{self.base_url}/workspaces"))
        workspaces = [Workspace.from_dict(workspace) for workspace in response["workspaces"]]
        for workspace in workspaces:
            workspace.set_client(self)
        return workspaces

    def create_workspace(self, workspace_request: WorkspaceRequest) -> Workspace:
        response = self._handle_response(self.session.post(
            f"{self.base_url}/workspaces",
            json=workspace_request.to_dict()
        ))
        workspace = Workspace.from_dict(response["workspace"])
        workspace.set_client(self)
        return workspace

    def update_workspace(self, workspace_id: str, workspace_request: WorkspaceRequest) -> Workspace:
        response = self._handle_response(self.session.put(
            f"{self.base_url}/workspaces/{workspace_id}",
            json=workspace_request.to_dict()
        ))
        workspace = Workspace.from_dict(response["workspace"])
        workspace.set_client(self)
        return workspace

    def get_tasks(self, workspace_id: str, statuses: Optional[str] = None) -> List[Task]:
        params = {"statuses": statuses} if statuses else {}
        response = self._handle_response(self.session.get(
            f"{self.base_url}/workspaces/{workspace_id}/tasks",
            params=params
        ))
        return [Task.from_dict(task) for task in response["tasks"]]

    def get_task(self, workspace_id: str, task_id: str) -> Task:
        response = self._handle_response(self.session.get(
            f"{self.base_url}/workspaces/{workspace_id}/tasks/{task_id}"
        ))
        return Task.from_dict(response["task"])

    def create_task(self, workspace_id: str, task_request: TaskRequest) -> Task:
        response = self._handle_response(self.session.post(
            f"{self.base_url}/workspaces/{workspace_id}/tasks",
            json=task_request.to_dict()
        ))
        return Task.from_dict(response["task"])

    def update_task(self, workspace_id: str, task_id: str, task_request: TaskRequest) -> Task:
        response = self._handle_response(self.session.put(
            f"{self.base_url}/workspaces/{workspace_id}/tasks/{task_id}",
            json=task_request.to_dict()
        ))
        return Task.from_dict(response["task"])

    def delete_task(self, workspace_id: str, task_id: str):
        self._handle_response(self.session.delete(
            f"{self.base_url}/workspaces/{workspace_id}/tasks/{task_id}"
        ))

# Usage example:
if __name__ == "__main__":
    # Initialize the client
    client = SidekickClient()

    # Get workspaces
    workspaces = client.get_workspaces()
    for workspace in workspaces:
        print(workspace.name)

    # create a workspace
    workspace = client.create_workspace(WorkspaceRequest(
        name="New workspace",
        local_repo_dir="path/to/repo"
    ))

    """
    # Create a new task
    new_task = workspace.create_task(TaskRequest(
        description="New task",
        status="pending",
        agent_type="default",
        flow_type="standard"
    ))

    # Get tasks for this workspace
    tasks = workspace.get_tasks()
    for task in tasks:
        print(task.description)

    # Update a task
    updated_task = workspace.update_task(new_task.id, TaskRequest(
        description="Updated task",
        status="in_progress",
        agent_type="default",
        flow_type="standard"
    ))

    # Delete a task
    workspace.delete_task(new_task.id)
    """
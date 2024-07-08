from copy import deepcopy
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Callable

from langchain_openai import ChatOpenAI
from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities import I18N, Converter, ConverterError, Printer
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser

class Task(BaseModel):
    """Class that represents a task to be executed."""

    class Config:
        arbitrary_types_allowed = True

    __hash__ = object.__hash__  
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    i18n: I18N = I18N()
    thread: Optional[threading.Thread] = None
    prompt_context: Optional[str] = None
    description: str = Field(description="Description of the actual task.")
    expected_output: str = Field(description="Clear definition of expected output for the task.")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for the task")
    callback: Optional[Callable[[TaskOutput], None]] = Field(default=None, description="Callback function post task completion.")
    agent: Optional[Agent] = Field(default=None, description="Agent responsible for executing the task.")
    context: Optional[List["Task"]] = Field(default=None, description="Context tasks for input.")
    async_execution: Optional[bool] = Field(default=False, description="Asynchronous execution flag.")
    output_json: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model for JSON output.")
    output_pydantic: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model for task output.")
    output_file: Optional[str] = Field(default=None, description="File path for storing output.")
    output: Optional[TaskOutput] = Field(default=None, description="Final output result post execution.")
    tools: Optional[List[Any]] = Field(default_factory=list, description="Tools available for task execution.")
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True, description="Unique identifier for the task.")
    human_input: Optional[bool] = Field(default=False, description="Flag for human review.")
    rci_iterations: int = Field(default=5, description='Number of iterations for RCI chain.')

    _original_description: str | None = None
    _original_expected_output: str | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", 
                "This field is not to be set by the user.", 
                {}
            )

    @field_validator("output_file")
    @classmethod
    def output_file_validation(cls, value: str) -> str:
        return value.lstrip('/')

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Task":
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def check_tools(self):
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    @model_validator(mode="after")
    def check_output(self):
        output_types = [self.output_json, self.output_pydantic]
        if sum(bool(t) for t in output_types) > 1:
            raise PydanticCustomError(
                "output_type",
                "Only one of output_pydantic or output_json can be set.",
                {}
            )
        return self

    def execute( 
        self,
        agent: Agent | None = None,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        agent = agent or self.agent
        if not agent:
            raise Exception(f"The task '{self.description}' has no specified agent.")

        if self.context:
            context_outputs = []
            for task in self.context:
                if task.async_execution and task.thread:
                    task.thread.join()
                if task and task.output:
                    context_outputs.append(task.output.raw_output)
            context = "\n".join(context_outputs)

        self.prompt_context = context
        tools = tools or self.tools

        if self.async_execution:
            self.thread = threading.Thread(target=self._execute, args=(agent, context, tools))
            self.thread.start()
            return "Async task started"
        else:
            return self._rci_chain()

    def _execute(self, agent: Agent, context: Optional[str], tools: Optional[List[Any]]) -> str:
        max_retries = 5
        delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                result = agent.execute_task(self, context, tools)
                exported_output = self._export_output(result)
                self.output = TaskOutput(
                    description=self.description,
                    exported_output=exported_output,
                    raw_output=result,
                    agent=agent.role,
                )
                if self.callback:
                    self.callback(self.output)
                return exported_output
            except openai.error.RateLimitError:
                delay *= 2  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            except Exception as e:
                raise e

    def prompt(self) -> str:
        tasks_slices = [self.description, self.i18n.slice("expected_output").format(expected_output=self.expected_output)]
        return "\n".join(tasks_slices)

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output

        if inputs:
            self.description = self._original_description.format(**inputs)
            self.expected_output = self._original_expected_output.format(**inputs)

    def increment_tools_errors(self) -> None:
        self.tools_errors += 1

    def increment_delegations(self) -> None:
        self.delegations += 1

    def copy(self) -> 'Task':
        exclude = {"id", "agent", "context", "tools"}
        copied_data = self.model_dump(exclude=exclude)
        copied_data = {k: v for k, v in copied_data.items() if v is not None}
        return Task(
            **copied_data,
            context=[t.copy() for t in self.context] if self.context else None,
            agent=self.agent.copy() if self.agent else None,
            tools=deepcopy(self.tools) if self.tools else None,
        )

    def _export_output(self, result: str) -> Any:
        instructions = "I'm gonna convert this raw text into valid JSON."
        model = self.output_pydantic or self.output_json
        exported_result = result

        if model:
            try:
                exported_result = model.model_validate_json(result)
                return exported_result.model_dump() if self.output_json else exported_result
            except Exception:
                match = re.search(r"({.*})", result, re.DOTALL)
                if match:
                    try:
                        exported_result = model.model_validate_json(match.group(0))
                        return exported_result.model_dump() if self.output_json else exported_result
                    except Exception:
                        pass

            llm = self.agent.function_calling_llm or self.agent.llm
            if not self._is_gpt(llm):
                schema = PydanticSchemaParser(model=model).get_schema()
                instructions += f"\n\nThe JSON should have the following structure:\n{schema}"

            converter = Converter(llm=llm, text=result, model=model, instructions=instructions)
            exported_result = converter.to_pydantic() if self.output_pydantic else converter.to_json()

            if isinstance(exported_result, ConverterError):
                Printer().print(f"{exported_result.message} Using raw output instead.", color="red")
                exported_result = result

        if self.output_file:
            content = exported_result if not self.output_pydantic else exported_result.json()
            self._save_file(content)
        return exported_result

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None

    def _save_file(self, content: Any) -> None:
        directory = os.path.dirname(self.output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(content)

    def __repr__(self):
        return f"Task(description={self.description}, expected_output={self.expected_output})"

    def _critique(self, question,output: str) -> bool:
        """Critique the given output to determine if a more accurate answer is possible."""
        # Store the original description to restore later
        original_description = self.description
        
        # Modify the description to ask for a more accurate answer
        critique_question = f"Can you give a more accurate answer for this question :{question} based on the current answer \nCurrent answer: \n{output} ?  \nStrictly answer in yes or no."
        self.description = critique_question

        # Execute the task with the modified description
        new_output = self._execute(self.agent,self.context,self.tools)
        
        
        # Check if the output indicates that a more accurate answer is possible
        if 'yes' in new_output.lower():
            # Modify the description to ask for an improved answer
            improvement_prompt = f"Provide an improved answer for this question: {question}\nCurrent answer: {output}"
            self.description = improvement_prompt
            return True
        else:
            # Reset the description to its original value
            self.description = question
            print("Can't find better result")
            return False

    def _rci_chain(self) -> str:
        """Execute a recursive critique improvement chain."""
        # Initialize the final output
        final_output = None
        question=self.description

        # Iterate for the specified number of iterations
        for itr in range(self.rci_iterations):
            # Log the iteration
            print(f'Starting iteration {itr + 1}/{self.rci_iterations}')

            # Execute the task and store the output
            output = self._execute(self.agent, self.context, self.tools)

            # Print the output for this iteration
            print(f'Iteration {itr + 1} output:\n{output}')

            # Check if the critique function indicates that a better result is possible
            if self._critique(question,output):
                answer_after_critique=self._execute(self.agent,self.description,self.tools)
                print(f'Answer after critique :\n{answer_after_critique}')
                # For that case in which we got improved answer every time
                final_output=answer_after_critique
                continue
            else:
                # Store the final output and exit the loop as no better result found
                final_output = output
                break

        # Log and return the final output
        print(f'Final output after RCI chain:\n{final_output}')
        return final_output


#This is a temporary check for testing
#Another temporary change

#Another change 
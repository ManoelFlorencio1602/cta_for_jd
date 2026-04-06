from pathlib import Path
import jinja2


class PromptLoader:
    def __init__(
            self,
            system_prompt_path: Path = Path("prompts/system_prompt.jinja2"),
            user_prompt_path: Path = Path("prompts/user_prompt.jinja2"),
    ):
        self.system_template = self._load_template(system_prompt_path)
        self.user_template = self._load_template(user_prompt_path)

    def system_prompt(self):
        return self.system_template.render()

    def user_prompt(
            self,
            query_table_name,
            query_table_description,
            query_table_column_names,
            query_table_column_descriptions,
            candidate_table_name,
            candidate_table_description,
            candidate_table_colunm_names,
            candidate_table_column_descriptions,
    ):
        return self.user_template.render(
            query_table_name=query_table_name,
            query_table_description=query_table_description,
            query_table_column_names=query_table_column_names,
            query_table_column_descriptions=query_table_column_descriptions,
            candidate_table_name=candidate_table_name,
            candidate_table_description=candidate_table_description,
            candidate_table_colunm_names=candidate_table_colunm_names,
            candidate_table_column_descriptions=candidate_table_column_descriptions,
        )


    def _load_template(self, template_path: Path) -> jinja2.Template:
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, "r") as f:
            template_str = f.read()

        return jinja2.Template(template_str)

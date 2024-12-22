from invoke.context import Context
from invoke.tasks import task


@task
def format(ctx: Context) -> None:
    """Run ruff to lint and format the code."""
    ctx.run("poetry run ruff format src/ tasks.py tests/")


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff to lint and format the code."""
    options = "--fix" if fix else ""
    ctx.run(f"poetry run ruff check src/ tasks.py tests/ {options}")


@task
def docs(ctx: Context) -> None:
    """Generate HTML documentation with pdoc."""
    ctx.run("poetry run pdoc src/telco-churn -o docs -d google")


@task
def check(ctx: Context) -> None:
    """Run type, lint and format checks"""
    lint(ctx)
    format(ctx)
    docs(ctx)

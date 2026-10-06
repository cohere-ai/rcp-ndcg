# Recipes

One directory per served model, written by the recipe lanes:

    recipes/<id>/recipe.yaml       # the Recipe schema (schema/recipe.schema.json in the package)
    recipes/<id>/template.jinja               # the chat template for vllm serve --chat-template, when needed
    recipes/<id>/reference.py                 # the reference, run as a subprocess (--reference-python; see docs)
    recipes/<id>/requirements-reference.txt   # optional: the reference environment, overriding the shared one

`<id>` matches `^[a-z0-9][a-z0-9.-]*$` and equals the directory name. See `docs/how-to/add-a-model.md` for the
field-by-field guide, and the tests' fixtures under `tests/fixtures/recipes/` for complete examples per role.

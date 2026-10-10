# Recipes

One directory per model family, written by the recipe authors (decision 34: one family, many sizes,
every size its own tested recipe id):

    recipes/<family>/family.yaml              # the Family schema (schema/family.schema.json in the package):
                                              # the shared blocks + the variants table
    recipes/<family>/template.jinja           # the family's ONE chat template for vllm serve --chat-template, when needed
    recipes/<family>/reference.py             # the family's ONE reference, run as a subprocess per variant
                                              # (--reference-python; see docs)
    recipes/<family>/requirements-reference.txt  # the family's reference environment (installed by the node's bootstrap)

`<family>` matches `^[a-z0-9][a-z0-9.-]*$` and equals the directory name; each variant id is a full recipe id
and is never a family id. See `docs/how-to/add-a-model.md` for the field-by-field guide, and the tests'
fixtures under `tests/fixtures/recipes/` for complete examples per role.

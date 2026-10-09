# Recipes

One directory per model family, written by the recipe lanes (owner decision 34: one family, many sizes):

    recipes/<family>/family.yaml                  # the Family schema: the shared blocks plus the variants table
    recipes/<family>/template.jinja               # the chat template for vllm serve --chat-template, when needed
    recipes/<family>/reference.py                 # the family's one reference, run as a subprocess (--reference-python; see docs)
    recipes/<family>/requirements-reference.txt   # optional: the reference's own environment (installed by the node's bootstrap)

`<family>` matches `^[a-z0-9][a-z0-9.-]*$` and equals the directory name; it is never served. Every variant's
`id` is the lowercased canonical Hub repo name of its checkpoint -- the served recipe id `serve`,
`recipe: <id>` and the wave lists take, resolving to the recipe schema in `schema/recipe.schema.json`. See
`docs/how-to/add-a-model.md` for the field-by-field guide, and the tests' fixtures under
`tests/fixtures/recipes/` for complete examples per role.

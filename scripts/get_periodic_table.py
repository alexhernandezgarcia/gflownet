from mendeleev.fetch import fetch_table
import json


def element_maps(table_name, table_key, elems, write=False):
    """
    Returns or writes to json file; map created from
    "table_name" dataframe in mendeleev, using
    column "atomic number" to column "table_key"

    Elems are either first n number entries to
    "table_name" or list of strings with element
    symbols for the "elements table"
    """
    base = fetch_table(table_name)
    if isinstance(elems, int):
        df = base.iloc[:elems]
    elif isinstance(elems, tuple):
        df = base.iloc[elems[0] - 1 : elems[1] - 1]
    elif isinstance(elems, list) and table_name == "elements":
        df = base[base[table_key].isin(elems)]

    if table_name == "oxidationstates":
        df = df.groupby("atomic_number")[table_key].apply(list).to_dict()
    elif table_name == "elements":
        df = df.set_index("atomic_number")[table_key].to_dict()

    if write:
        fptr = table_key + ".json"
        with open(fptr, "w", encoding="utf-8") as fobj:
            json.dump(df, fobj, ensure_ascii=False, indent=4)
    else:
        print(df)


if __name__ == "__main__":
    element_maps("oxidationstates", "oxidation_state", 10)

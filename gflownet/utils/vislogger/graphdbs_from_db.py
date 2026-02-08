"""Utilities for the logger to build the graph db from the trajectory data."""


def create_tables(conn):
    """Create both nodes and edges tables in the same database."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            node_type TEXT,
            reward REAL
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT,
            source TEXT,
            target TEXT,
            trajectory_id INTEGER,
            iteration INTEGER,
            logprobs_forward REAL,
            logprobs_backward REAL
        )
    """
    )
    conn.commit()


def insert_root_node(conn):
    """Add the root."""
    conn.execute(
        """
        INSERT OR IGNORE INTO nodes (id, node_type, reward)
        VALUES ('#', 'start', NULL)
    """
    )
    conn.commit()


def create_indexes(conn):
    """Create indexes for efficient truncation queries."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
    conn.commit()


def truncate_graph(conn):
    """Truncate the graph by removing nodes with one parent and one child."""
    cursor = conn.cursor()

    # Step 1: Identify removable nodes
    # A node is removable if it has exactly 1 unique predecessor AND 1 unique successor
    cursor.execute(
        """
            CREATE TEMP TABLE IF NOT EXISTS removable_nodes AS
            WITH node_connections AS (
                SELECT
                    n.id,
                    COUNT(DISTINCT e_in.source) as num_predecessors,
                    COUNT(DISTINCT e_out.target) as num_successors,
                    n.node_type AS type
                FROM nodes n
                LEFT JOIN edges e_in ON n.id = e_in.target
                LEFT JOIN edges e_out ON n.id = e_out.source
                GROUP BY n.id
            )
            SELECT id
            FROM node_connections
            WHERE num_predecessors = 1 AND num_successors = 1 AND type != "final"
        """
    )

    removable_count = cursor.execute("SELECT COUNT(*) FROM removable_nodes").fetchone()[
        0
    ]

    if removable_count == 0:
        return

    # Step 2: Create lookup table mapping each removable node to its unique successor
    cursor.execute(
        """
            CREATE TEMP TABLE IF NOT EXISTS node_successors AS
            SELECT DISTINCT
                e.source as node_id,
                e.target as successor_id
            FROM edges e
            WHERE e.source IN (SELECT id FROM removable_nodes)
        """
    )

    # Step 3: Iteratively bypass removable nodes
    iteration = 0
    max_iterations = 1000  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        cursor.execute("DROP TABLE IF EXISTS edges_to_delete")

        # Materialize edges to update/delete
        cursor.execute(
            """
            CREATE TEMP TABLE edges_to_delete AS
            SELECT e.source, e.target, e.trajectory_id
            FROM edges e
            WHERE e.source NOT IN (SELECT id FROM removable_nodes)
              AND e.target IN (SELECT id FROM removable_nodes)
        """
        )

        cursor.execute("SELECT COUNT(*) FROM edges_to_delete")
        edges_to_update = cursor.fetchone()[0]

        if edges_to_update == 0:
            cursor.execute("DROP TABLE edges_to_delete")
            break

        # Insert bypass edges
        cursor.execute(
            """
            INSERT INTO edges (
                id,
                source,
                target,
                trajectory_id,
                iteration,
                logprobs_forward,
                logprobs_backward
            )
            SELECT
                e.id,
                e.source,
                ns.successor_id AS target,
                e.trajectory_id,
                e.iteration,
                e.logprobs_forward + COALESCE(e_next.logprobs_forward, 0),
                e.logprobs_backward + COALESCE(e_next.logprobs_backward, 0)
            FROM edges e
            INNER JOIN edges_to_delete etd
                ON e.source = etd.source
               AND e.target = etd.target
               AND e.trajectory_id = etd.trajectory_id
            INNER JOIN node_successors ns
                ON e.target = ns.node_id
            LEFT JOIN edges e_next
                ON e_next.source = e.target
               AND e_next.target = ns.successor_id
               AND e_next.trajectory_id = e.trajectory_id
        """
        )

        # Delete only the original edges
        cursor.execute(
            """
            DELETE FROM edges
            WHERE (source, target, trajectory_id) IN (
                SELECT source, target, trajectory_id
                FROM edges_to_delete
            )
        """
        )

        cursor.execute("DROP TABLE edges_to_delete")
        conn.commit()

    if iteration >= max_iterations:
        print(
            f"Warning: Vislogger Building Graph DB reached maximum iterations "
            f"({max_iterations})"
        )

    # Step 4: Handle edges between removable nodes (internal chain edges)
    cursor.execute(
        """
            DELETE FROM edges
            WHERE source IN (SELECT id FROM removable_nodes)
               OR target IN (SELECT id FROM removable_nodes)
        """
    )

    # Step 6: Clean up temporary tables
    cursor.execute("DROP TABLE IF EXISTS removable_nodes")
    cursor.execute("DROP TABLE IF EXISTS node_successors")

    conn.commit()


def create_graph_dbs(conn):
    """Create the graph db."""
    create_tables(conn)
    insert_root_node(conn)

    read_cur = conn.cursor()  # Cursor for reading
    write_cur = conn.cursor()  # Cursor for writing

    seen_nodes = set(["#"])

    read_cur.execute(
        """
        SELECT
            final_id,
            step,
            final_object,
            text,
            iteration,
            total_reward,
            logprobs_forward,
            logprobs_backward
        FROM trajectories
        ORDER BY final_id, step
    """
    )

    current_trajectory = None
    prev_row = None

    for row in read_cur:
        (
            trajectory_id,
            step,
            final_object,
            text,
            iteration,
            total_reward,
            logpf,
            logpb,
        ) = row

        # --- new trajectory starts ---
        if trajectory_id != current_trajectory:
            current_trajectory = trajectory_id
            prev_row = None

        # --- insert or update node ---
        if text not in seen_nodes:
            node_type = "final" if final_object == 1 else "standard"
            reward = total_reward if final_object == 1 else None

            write_cur.execute(
                """
                INSERT OR IGNORE INTO nodes (id, node_type, reward)
                VALUES (?, ?, ?)
            """,
                (text, node_type, reward),
            )

            seen_nodes.add(text)
        elif final_object == 1:
            # Node exists but this is a final occurrence - update it
            write_cur.execute(
                """
                UPDATE nodes
                SET node_type = 'final', reward = ?
                WHERE id = ?
            """,
                (total_reward, text),
            )

        # --- create edge ---
        edge_id = f"{trajectory_id}_{step}"

        if prev_row is None:
            # root -> first node
            write_cur.execute(
                """
                INSERT INTO edges (
                    id,
                    source,
                    target,
                    trajectory_id,
                    iteration,
                    logprobs_forward,
                    logprobs_backward
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (edge_id, "#", text, trajectory_id, iteration, logpf, logpb),
            )
        else:
            prev_text = prev_row[3]

            write_cur.execute(
                """
                INSERT INTO edges (
                    id,
                    source,
                    target,
                    trajectory_id,
                    iteration,
                    logprobs_forward,
                    logprobs_backward
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (edge_id, prev_text, text, trajectory_id, iteration, logpf, logpb),
            )

        prev_row = row

    conn.commit()

    # Create indexes for efficient truncation
    create_indexes(conn)

    # Perform truncation
    truncate_graph(conn)

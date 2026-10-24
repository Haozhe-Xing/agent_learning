"""Real SQLite checks: authorization, resource limits, and unchanged source data."""
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import tempfile
import unittest

from reference_code.sqlite_readonly import ClarificationRequired, QueryBudgetExceeded, ReadOnlySQLite, TextToSQL


class ReadOnlySQLiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sales #1?.db'
        with sqlite3.connect(self.path) as connection:
            connection.executescript('CREATE TABLE orders(id INTEGER PRIMARY KEY, amount INTEGER);'
                                     'CREATE TABLE private_notes(note TEXT);')
            connection.executemany('INSERT INTO orders(amount) VALUES (?)', [(n,) for n in range(100)])
            connection.execute("INSERT INTO private_notes VALUES ('not authorized')")
        self.database = ReadOnlySQLite(self.path, allowed_tables={'orders'})

    def test_query_and_parameters_use_real_database(self):
        result = self.database.execute_readonly('SELECT sum(amount) AS total FROM orders WHERE id <= ?', (3,))
        self.assertEqual(result, [{'total': 3}])

    def test_schema_contains_only_allowed_tables_and_no_sample_rows(self):
        schemas = self.database.get_table_schemas()
        self.assertEqual(set(schemas), {'orders'})
        self.assertEqual(schemas['orders']['sample_data'], [])

    def test_empty_allowlist_does_not_restore_default_access(self):
        database = ReadOnlySQLite(self.path, allowed_tables=set())
        with self.assertRaises(PermissionError):
            database.execute_readonly('SELECT * FROM orders')

    def test_write_attach_pragma_and_extension_actions_are_rejected(self):
        statements = ['DELETE FROM orders', 'DROP TABLE orders',
                      "ATTACH DATABASE ':memory:' AS extra", 'PRAGMA query_only=OFF',
                      "SELECT load_extension('missing')", 'SELECT randomblob(1000000)']
        for statement in statements:
            with self.subTest(sql=statement), self.assertRaises(PermissionError):
                self.database.execute_readonly(statement)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM orders').fetchone()[0], 100)

    def test_other_tables_are_denied_even_inside_subqueries(self):
        with self.assertRaises(PermissionError):
            self.database.execute_readonly('SELECT (SELECT note FROM private_notes) AS value FROM orders LIMIT 1')

    def test_keywords_inside_data_are_not_security_decisions(self):
        self.assertEqual(self.database.execute_readonly("SELECT 'DROP TABLE' AS example"),
                         [{'example': 'DROP TABLE'}])

    def test_multiple_statements_do_not_run(self):
        with self.assertRaises(sqlite3.ProgrammingError):
            self.database.execute_readonly('SELECT 1; DELETE FROM orders')

    def test_result_limit_is_applied_before_unbounded_fetch(self):
        with self.assertRaisesRegex(ValueError, 'exceeds 3 rows'):
            ReadOnlySQLite(self.path, {'orders'}, max_rows=3).execute_readonly('SELECT * FROM orders')

    def test_expensive_cross_join_is_interrupted(self):
        database = ReadOnlySQLite(self.path, {'orders'}, max_vm_steps=500)
        with self.assertRaises(QueryBudgetExceeded):
            database.execute_readonly('SELECT count(*) FROM orders a, orders b, orders c')

    def test_connection_itself_is_readonly(self):
        with self.database.get_connection() as connection, self.assertRaises(sqlite3.OperationalError):
            connection.execute('INSERT INTO orders(amount) VALUES (999)')

    def test_missing_database_is_not_created(self):
        absent = self.path.with_name('absent.db')
        with self.assertRaises(FileNotFoundError):
            ReadOnlySQLite(absent)
        self.assertFalse(absent.exists())

    def test_invalid_limits_and_unknown_tables_fail_before_query(self):
        for options in ({'max_rows': 0}, {'max_vm_steps': True}, {'timeout_seconds': float('inf')},
                        {'allowed_tables': {'missing'}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ReadOnlySQLite(self.path, **options)

    def test_duplicate_column_names_require_explicit_aliases(self):
        with self.assertRaisesRegex(ValueError, 'column aliases'):
            self.database.execute_readonly('SELECT 1 AS value, 2 AS value')

    def test_case_conflicting_aliases_do_not_silently_duplicate_values(self):
        with self.assertRaisesRegex(ValueError, 'column aliases'):
            self.database.execute_readonly('SELECT 1 AS value, 2 AS VALUE')
        self.assertEqual(self.database.execute_readonly('SELECT 1 AS first, 2 AS second'),
                         [{'first': 1, 'second': 2}])


class TextToSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.db'
        with sqlite3.connect(self.path) as connection:
            connection.execute('CREATE TABLE orders(amount INTEGER)')
            connection.execute('INSERT INTO orders VALUES (25)')
        self.database = ReadOnlySQLite(self.path, {'orders'})

    async def test_fixed_provider_and_host_execute_separately(self):
        class Provider:
            async def ainvoke(self, messages):
                self.messages = messages
                return SimpleNamespace(content='SELECT sum(amount) AS total FROM orders')
        provider = Provider()
        query = await TextToSQL(provider, self.database).convert('Total amount?', as_of='2026-03-01')
        self.assertEqual(self.database.execute_readonly(query), [{'total': 25}])
        self.assertNotIn('25', provider.messages[-1][1])  # Schema contains no sample rows.
        self.assertIn('2026-03-01', provider.messages[-1][1])

    async def test_ambiguous_question_requires_clarification(self):
        class Provider:
            async def ainvoke(self, messages):
                return SimpleNamespace(content='NEEDS_CLARIFICATION')
        with self.assertRaises(ClarificationRequired):
            await TextToSQL(Provider(), self.database).convert('Large orders?', as_of='2026-03-01')

    async def test_generated_write_still_cannot_execute(self):
        class Provider:
            async def ainvoke(self, messages):
                return SimpleNamespace(content='DELETE FROM orders')
        query = await TextToSQL(Provider(), self.database).convert('Delete orders', as_of='2026-03-01')
        with self.assertRaises(PermissionError):
            self.database.execute_readonly(query)

    async def test_invalid_model_response_is_not_a_query(self):
        class Provider:
            async def ainvoke(self, messages):
                return SimpleNamespace(content=['not a plain SQL string'])
        with self.assertRaises(ValueError):
            await TextToSQL(Provider(), self.database).convert('Total?', as_of='2026-03-01')


if __name__ == '__main__':
    unittest.main()

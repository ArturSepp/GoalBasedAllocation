# Replication inputs

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The four CSV files are existing snapshots moved byte-for-byte from this
study's former `data/` directory. Their original provenance and dates remain in
the study README and file headers. The SHA-256 manifest records this preserved
bundle with CRLF normalized to LF, matching Git's existing text policy. The local
migration preserves original on-disk line endings as well. Existing tracking does
not establish general vendor-data redistribution
rights or approve further snapshots. The fetcher is now `../fetch_option_chain.py`;
new fetched data goes to external runtime storage.

Keep restricted additional inputs in ignored `local/`. Do not overwrite these
inputs with new runs.

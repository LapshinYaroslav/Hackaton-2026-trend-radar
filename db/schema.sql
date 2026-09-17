-- Черновик схемы БД.
-- Файл монтируется в /docker-entrypoint-initdb.d/ и выполняется
-- только при ПЕРВОМ запуске Postgres на пустом volume.
-- После изменения схемы: docker compose down -v && docker compose up -d

-- Пока заглушка: полноценные таблицы добавит команда (api/db) через PR.
SELECT 1;

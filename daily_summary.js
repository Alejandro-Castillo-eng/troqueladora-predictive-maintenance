(function (root) {
    'use strict';

    const sensors = [
        { key: 'temp', name: 'Temperatura del motor', unit: '°C' },
        { key: 'pres', name: 'Presión hidráulica', unit: 'bar' },
        { key: 'vib', name: 'Vibración del troquel', unit: 'mm/s' },
        { key: 'corr', name: 'Corriente del motor', unit: 'A' }
    ];

    function buildReport(rows, classify) {
        const states = { NORMAL: 0, ADVERTENCIA: 0, CRITICO: 0 };
        const statistics = sensors.map(sensor => {
            const values = rows.map(row => row[sensor.key]).filter(Number.isFinite);
            return {
                ...sensor,
                count: values.length,
                mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
                min: values.length ? Math.min(...values) : null,
                max: values.length ? Math.max(...values) : null
            };
        });
        for (const row of rows) states[classify(row).estado] += 1;
        const rulValues = rows.map(row => row.rul).filter(Number.isFinite);
        const first = rows[0];
        const last = rows[rows.length - 1];
        return {
            date: last.date,
            count: rows.length,
            firstTime: first.hm,
            lastTime: last.hm,
            lastTimestamp: last.t,
            states,
            sensors: statistics,
            lastRul: Number.isFinite(last.rul) ? last.rul : null,
            minimumRul: rulValues.length ? Math.min(...rulValues) : null
        };
    }

    function createTracker() {
        const completed = new Map();
        let latest = null;
        return {
            advance(sample, rows, classify, date) {
                let generated = false;
                // An 80-row scenario ending is not necessarily the end of its day.
                const atDayEnd = Boolean(sample && rows.length && sample.t === rows[rows.length - 1].t);
                if (atDayEnd) {
                    if (!completed.has(sample.date)) {
                        completed.set(sample.date, buildReport(rows, classify));
                        generated = true;
                    }
                    latest = completed.get(sample.date);
                }
                const current = sample ? rows.findIndex(row => row.t === sample.t) + 1 : 0;
                return {
                    generated,
                    atDayEnd,
                    date: sample ? sample.date : date,
                    current,
                    total: rows.length,
                    report: latest
                };
            }
        };
    }

    const tracker = createTracker();
    let renderedDate = null;
    const formatDate = date => date ? date.split('-').reverse().join('/') : '—';
    const number = value => Number.isFinite(value)
        ? value.toLocaleString('es-PE', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—';

    function addText(parent, tag, text, className) {
        const element = document.createElement(tag);
        element.textContent = text;
        if (className) element.className = className;
        parent.appendChild(element);
        return element;
    }

    function renderReport(report, parent) {
        parent.replaceChildren();
        addText(parent, 'p', `Último día completado: ${formatDate(report.date)}`, 'daily-summary-date');
        addText(parent, 'p', `${report.count} tomas · ${report.firstTime} a ${report.lastTime}`, 'daily-summary-muted');
        addText(parent, 'h3', 'Comportamiento del día');
        const { NORMAL: normal, ADVERTENCIA: warning, CRITICO: critical } = report.states;
        const description = critical > 0
            ? `El panel clasificó ${critical} tomas como críticas y ${warning} como advertencia.`
            : warning > 0
                ? `El panel clasificó ${warning} tomas como advertencia; no hubo tomas clasificadas como críticas.`
                : 'Todas las tomas se clasificaron como normales según las reglas del panel.';
        addText(parent, 'p', description);
        addText(parent, 'p', `Normal: ${normal} · Advertencia: ${warning} · Crítico: ${critical}`, 'daily-summary-muted');
        addText(parent, 'h3', 'Sensores');
        for (const sensor of report.sensors) {
            const card = addText(parent, 'div', '', 'daily-summary-sensor');
            addText(card, 'h4', sensor.name);
            addText(card, 'p', `Promedio: ${number(sensor.mean)} ${sensor.unit}`);
            addText(card, 'p', `Mín.: ${number(sensor.min)} · Máx.: ${number(sensor.max)} ${sensor.unit}`, 'daily-summary-muted');
            if (sensor.count < report.count) addText(card, 'p', `${sensor.count} lecturas válidas de ${report.count}.`, 'daily-summary-muted');
        }
        addText(parent, 'h3', 'RUL estimado');
        addText(parent, 'p', `Al cierre: ${number(report.lastRul)} días operativos.`);
        addText(parent, 'p', `Mínimo del día: ${number(report.minimumRul)} días operativos.`, 'daily-summary-muted');
        addText(parent, 'p', '1 día operativo = 10 horas. Las alertas del panel y el RUL estimado no confirman una falla ocurrida.', 'daily-summary-note');
    }

    function update(sample, days, classify, date) {
        const rows = days.get(sample ? sample.date : date) || [];
        const state = tracker.advance(sample, rows, classify, date);
        const progress = document.getElementById('dailySummaryProgress');
        const content = document.getElementById('dailySummaryContent');
        const pending = document.getElementById('dailySummaryPending');
        if (!progress || !content || !pending) return;
        progress.textContent = state.total
            ? `${formatDate(state.date)} · Toma ${state.current} de ${state.total}${state.atDayEnd ? ' · Día completado' : ' · Resumen pendiente'}`
            : `${formatDate(state.date)} · Sin tomas registradas`;
        pending.hidden = Boolean(state.report);
        content.hidden = !state.report;
        // Keep the completed day's report visible when playback advances to the next day.
        if (state.report && renderedDate !== state.report.date) {
            renderReport(state.report, content);
            renderedDate = state.report.date;
        }
    }

    const api = { buildReport, createTracker, update };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.DailySummary = api;
}(typeof window !== 'undefined' ? window : globalThis));

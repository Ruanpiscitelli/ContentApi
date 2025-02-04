// Função modificada para gerenciar pontos de dados
function updateChart(chart, newData) {
    // Limitar o número de pontos mantidos
    chart.data.labels.push(new Date().toLocaleTimeString());
    if (chart.data.labels.length > MAX_DATA_POINTS) {
        chart.data.labels.shift();
    }
    chart.data.datasets.forEach((dataset) => {
        dataset.data.push(newData.value);
        if (dataset.data.length > MAX_DATA_POINTS) {
            dataset.data.shift(); // Remove o ponto mais antigo
        }
    });
    
    chart.update();
} 

// Configuração inicial do gráfico
options: {
    animation: false, // Desativa animações pesadas
    spanGaps: true,   // Melhora performance com dados faltantes
    elements: {
        point: {
            radius: 0 // Remove pontos visuais
        }
    }
} 
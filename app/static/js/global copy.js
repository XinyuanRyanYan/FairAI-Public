/**
 * Global functions
 */

/*[ [[true_positive, False_positive], [False_negative, true_negative]],
        [[true_positive, False_positive], [False_negative, true_negative]] ], */
// [[[1, 10], [1, 12]], [[1, 20], [10, 15]]]
// let confusionMatrixData = '';
let CFScale = '';   // the scale of confusion matrix (number->color)
let opacityScale = '';  // the scale of opacity and the color
let trainData = '';     // [[male_positive, male_negative], [female_positive, female_negative]]
let testData = '';
let weights = '';   // the weights of after reweighing
let LRFTrainData = '';      // the train data after the LRF
let OptimPreprocTrainData = ''  // the train data after the OptimPreproc

let outputDict = {'German': ['Approve', 'Deny'], 'Bank': ['Yes', 'No']};
let outputLabel = outputDict['German'];
let attrValueDict = {'German': ['Male', 'Female'], 'Bank': ['Married', 'Unmarried']};
let attrVs = attrValueDict['German'];

// the dimension of the confusion matrix div
let CMDim = {       // dimension for the confusion matrix div
    margin: {left: 30, right: 10, top: 35, bottom: 10},
    width: 180,
    height: 140
}
CMDim.wrapperWid = CMDim.width + CMDim.margin.left + CMDim.margin.right;
CMDim.wrapperHei = CMDim.height + CMDim.margin.top + CMDim.margin.bottom;

// the dimesion for the output div
let outputDivDim = {
    divMargin: {       // the margin of the the div containing the confusion matrix div
        top: 10,    // 30
        bottom: 15, // 30
        right: 10,      //15
        left: 10,       // 15
    },

    divGap: 10,       // the gap between the two divs

    accuracyHeight: 0,
}
outputDivDim.width = CMDim.wrapperWid + outputDivDim.divMargin.left + outputDivDim.divMargin.right;
outputDivDim.CMHeight = CMDim.wrapperHei + outputDivDim.divMargin.top + outputDivDim.divMargin.bottom;
outputDivDim.height = outputDivDim.CMHeight*2 + outputDivDim.accuracyHeight + outputDivDim.divGap*2;


// the dimension for the train/div panel
let ttDivDim = {
    barStyles: {        // the style of the bar chart
        barWidth: 25,
        innerPadding: 20, 
        outerPadding: 15
    },
    margin:{        // the margin for the svg
        top: 25,
        bottom: 25,
        right: 15,
        left: 15,
    }
}

ttDivDim.boundedWidth = ttDivDim.barStyles.outerPadding*2 + ttDivDim.barStyles.innerPadding + ttDivDim.barStyles.barWidth*4;
ttDivDim.wrapperWidth = ttDivDim.boundedWidth + ttDivDim.margin.left + ttDivDim.margin.right;
// the X & Y axis scale for the bar chars
let yScaleBarDict = {'German': d3.scaleLinear().domain([0, 400]).range([0, 600]), 'Bank': d3.scaleLinear().domain([0, 3000]).range([0, 500])};
let yScaleBar = yScaleBarDict['German'];   // have scalability issue
// let yScaleBar = d3.scaleLinear().domain([0, 3000]).range([0, 500]);   // have scalability issue
let xScaleBar = (i)=> ttDivDim.barStyles.outerPadding + ttDivDim.barStyles.barWidth * (2*i + 1) 
    + i * ttDivDim.barStyles.innerPadding;

// let attrVs = ['Male', 'Female'];

// the information of the four fairness metrics
let fairMetricInfo = {
    'SPD': {fullName: 'Statistical Parity Difference', range: [-1, 1], fair: 0},
    'DI': {fullName: 'Disparate Impact', range: [-0.5, 2.5], fair: 1},
    'EOD': {fullName: 'Equal Opportunity Difference', range: [-1, 1], fair: 0},
    'AOD': {fullName: 'Average Odds Difference', range: [-1, 1], fair: 0}
};
// the dimension for the fairness metric panel
let fairMetricDim = {
    margin: {left: 90, right: 20, top: 80, bottom: 30},
    wid: 310,
    itemHei: 35,    // the height of each fairness metric item
}
fairMetricDim.innerWid = fairMetricDim.wid - fairMetricDim.margin.left - fairMetricDim.margin.right;


/**
 * add a legend group in this svg
 */
function addLegend(svgSelector, text=''){
    let legendG = svgSelector.append('g');
    let rectWid = 7;
    let y = 0;

    let visLengend = (x, text)=>{
        let textX = x+rectWid+3;

        legendG.append('rect')
            .attr('x', x).attr('y', y).attr('width', rectWid).attr('height', rectWid)
            .attr('stroke', 'none')
            .attr('fill', text==outputLabel[0]? colorMap.orange : colorMap.blue);

        legendG.append('text')
            .attr('x', textX).attr('y', y+rectWid)
            .attr('class', 'fontFamily ticks')
            .text(text);
    }
    visLengend(0, outputLabel[0]);
    visLengend(55, outputLabel[1]);

    legendG.append('text')
        .attr('x', 98).attr('y', y+rectWid)
        .attr('class', 'fontFamily ticks')
        .text(text);

    return legendG;
}


/*
Visualize the output result including the 1/2 plots and a accuracy panel
{
        'data': [ [[true_positive, False_positive], [False_negative, true_negative]],
        [[true_positive, False_positive], [False_negative, true_negative]] ],

        'accuracy': 0.98,

        'attrVs': ['Male', 'Female']  # sensitive value
    }
*/
function visOutputPanel(divSelector, jsonData, confusionMatrixData){

    let margin = 10;
    let legend_height = 20;
    
    let div_width = parseInt(divSelector.style("width"));
    let div_height = parseInt(divSelector.style("height"));

    if(divSelector.attr("id")){
        let divSelector_dom = document.getElementById(divSelector.attr("id"));
        div_width = divSelector_dom.clientWidth;
        div_height = divSelector_dom.clientHeight;
    }

    div_width = div_width  - 2*margin;
    div_height = (div_height - 3*margin - legend_height)/2;

    console.log("output", div_width)

    let maleDivSelector = divSelector.append('div')
    .classed("output-container", true)
    .style('top', '0')
    .style('left', `${margin}px`)
        .style('width', `${div_width}px`)
        .style('height', `${div_height}px`)
        .style('position', 'absolute')
        .attr("id", divSelector.attr("id")+"-male");
    let femaleDivSelector = divSelector.append('div').style('top', `${div_height+margin}px`)
        .classed("output-container", true)
        .style('left', '10px')
        .style('width', `${div_width}px`)
        .style('height', `${div_height}px`)
        .style('position', 'absolute')
        .attr("id", divSelector.attr("id")+"-female");

    // the accuracy panel
    // let accuracySelector = divSelector.append('div')
    //     .classed('accuracyPanel', true);

    // visualize the two confusion matrix
    // visConfusionMatrixPanel(maleDivSelector.append('div'), 0);
    // visConfusionMatrixPanel(femaleDivSelector.append('div'), 1);
    visConfusionMatrixPanel(maleDivSelector, confusionMatrixData, 0);
    visConfusionMatrixPanel(femaleDivSelector, confusionMatrixData, 1);

    // visualize the accuracy
    // const f = d3.format(",.1%");
    // accuracySelector.append('p').text(`Accuracy: ${f(accuracy)}`);
}

/* init the scale of confusion matrix */
function initCFScale(confusionMatrixData){
    // get the domain of the confusion matrix
    let min = 0, max = 0
    confusionMatrixData.forEach(ele=>{
        ele.forEach(e=>{
            max = d3.max(e)>max? d3.max(e): max;
        })
    })
    max = (parseInt(max/10)+1)*10;
    
    CFScale = d3.scaleLinear().domain([min,max])
        .range(['rgba(91, 155, 213, 0.2)', 'rgba(91, 155, 213, 1)'])
        // .range(['#DAE3F2', '#5B9BD5']);
    
    opacityScale = d3.scaleLinear().domain([min,max])
        .range([0.2, 1]);
    
}

/* visualize the color legend for the confunsion matrix 
 * @param {*} id: the id of gradient
 * horizontal: the direction of the color map
*/
function addColorLegend(svgSelector, gradId, horizontal=true){
    let legendG = svgSelector.append('g');
    let domain = CFScale.domain();  // the domain of the scale
    let startColor = CFScale(domain[0]);
    let endColor = CFScale(domain[1]);

    // if it is horizontal
    if(horizontal){
        let svg_width = parseInt(svgSelector.style("width"));
        let svg_height = parseInt(svgSelector.style("height"));
        let margin = svg_height/4;
        let legend_width = svg_width-10;
        let legend_height = Math.min(10, svg_height - 2*margin);
        let dim = {
            wid: legend_width,
            hei: legend_height,
        };
        // gradient generator
        let graGenerator = svgSelector.append('linearGradient').attr('id', gradId)
            .attr('x1', '0').attr('x2', '1').attr('y1', '0').attr('y2', '0');
        graGenerator.append('stop').attr('offset', '0').attr('stop-color', startColor);
        graGenerator.append('stop').attr('offset', '1').attr('stop-color', endColor);

        // rect
        legendG.append('rect').attr('x', margin).attr('y', margin).attr('width', dim.wid).attr('height', dim.hei).attr('fill', `url(#${gradId})`);
        // dismiss ticks, use the start and the end number instead
        legendG.append('text')
            .attr('x', margin).attr('y', margin).attr('dy', '-0.2em').attr('text-anchor', 'begin')
            .attr('font-size', '8px')
            .text(domain[0]);
        legendG.append('text')
            .attr('x', dim.wid).attr('y', margin).attr('dy', '-0.2em').attr('text-anchor', 'end')
            .attr('font-size', '8px')
            .text(domain[1]);

    }
    else{
        let svg_width = parseInt(svgSelector.style("width"));
        let svg_height = parseInt(svgSelector.style("height"));
        let margin_x = svg_width/3;
        let margin_y = 20;
        let legend_width = Math.min(10, svg_width - 2*margin_x)
        let legend_height = svg_height-2*margin_y;
        let dim = {
            wid: legend_width,
            hei: legend_height
        };
        // gradient generator
        let graGenerator = svgSelector.append('linearGradient').attr('id', gradId)
            .attr('x1', '0').attr('x2', '0').attr('y1', '1').attr('y2', '0');
        graGenerator.append('stop').attr('offset', '0').attr('stop-color', startColor);
        graGenerator.append('stop').attr('offset', '1').attr('stop-color', endColor);

        // rect
        legendG.append('rect').attr('x', margin_x).attr('y', margin_y).attr('width', dim.wid).attr('height', dim.hei).attr('fill', `url(#${gradId})`);

        // only add the minimum number and the maximum number
        legendG.append('text')
            .attr('x', margin_x+dim.wid/2).attr('y', margin_y)
            .attr('dy', '-0.2em')
            .attr('text-anchor', 'middle')
            .attr('font-size', '8px')
            .text(domain[1]);
        legendG.append('text')
            .attr('x', margin_x+dim.wid/2).attr('y', margin_y+dim.hei)
            .attr('dy', '1.1em')
            .attr('text-anchor', 'middle')
            .attr('font-size', '8px')
            .text(domain[0]);
    }
    
    
    return legendG;
}   

/**
 * visualize different kinds of fairness metrics 
 * @param {*} divSelector the fairness metric div
 * @param {*} metricName 
 * @param {*} metricData [{'original': 0.1}, {'mitigate': 0.2}...]
 */
function visFairMetricPanel(divSelector, metricName, metricData){
    // metricName = 'SPD';
    // metricData = [{'baseline': 0.1}, {'mitigate': 0.2}];
    // metricData.push({'reweigh': 0.2})
    // metricData.push({'post': -0.1})

    // set the class
    divSelector.selectAll('*').remove();
    divSelector.style('height', null);
    divSelector.classed('fairMetricDiv', true);    // reset the name here
    // reset the height
    let barLen = 50;
    let heightTemp = parseInt(divSelector.style('height'));
    divSelector.style('height', `${metricData.length==1? heightTemp-barLen:heightTemp}px`);

    divSelector.property('name', metricName);   // the name of this div selector

    // basic info
    let fairValue = fairMetricInfo[metricName].fair;
    let range = fairMetricInfo[metricName].range;
    let fontColor = '#213547';
    let gap = 10;       // the gap between two elements
    let baisAreaColor = '#EDEDED';          // '#FCF0F0'


    // init the div and the svg 
    // let margin_x = parseInt(divSelector.style("margin-left"));
    let margin_y = 10;
    let svg_width = parseInt(divSelector.style("width"));
    let svg_height = parseInt(divSelector.style("height"));
    let innerWid = svg_width - fairMetricDim.margin.left - fairMetricDim.margin.right;
    let itemHei = (svg_height - fairMetricDim.margin.top - fairMetricDim.margin.bottom)/metricData.length;
    // let barLen = itemHei;

    let divSvg = divSelector.append('svg')
        .attr('width', svg_width)
        .attr('height', svg_height);
    
    // visualization part
    let XScale = d3.scaleLinear()       // the scale for the metric value 
        .domain(fairMetricInfo[metricName].range)
        .range([fairMetricDim.margin.left, fairMetricDim.margin.left+innerWid]);

    // visualize the title
    divSvg.append('text').attr('x', svg_width/2).attr('y', 25)
        .attr('font-size', 15)
        .attr('font-weight', 500)
        .attr('fill', fontColor)
        .attr('text-anchor', 'middle')
        .text(fairMetricInfo[metricName].fullName);
    
    // visualize the bais part
    divSvg.append('rect')
        .attr('x', XScale(range[0])).attr('y', fairMetricDim.margin.top)
        .attr('width', XScale(fairValue)-XScale(range[0])).attr('height', metricData.length*barLen)
        .attr('fill', baisAreaColor)
        .attr('fill-opacity', '0.5');

    // viusualize the axis
    let tickSize = 5;
    let xAxis = d3.axisBottom(XScale).ticks(5).tickSize(tickSize);
    let axisG = divSvg.append('g')
        .attr('transform', `translate(0, ${fairMetricDim.margin.top})`)
        .call(xAxis);
    axisG.selectAll('line').attr('y2', -tickSize);      // reverse the ticks
    axisG.selectAll('text').attr('y', -15);         // change the y text
    axisG.select('path').remove();      // remove the previous one
    axisG.selectAll('line').attr('stroke', fontColor);
    axisG.selectAll('text').attr('fill', fontColor);
    axisG.append('line')
        .attr('x1', fairMetricDim.margin.left).attr('y1', 0)
        .attr('x2', fairMetricDim.margin.left+innerWid).attr('y2', 0)
        .attr('stroke', fontColor);
    

    // visualize the fair text & Line
    divSvg.append('text').attr('x', fairMetricDim.margin.left+innerWid/2).attr('y', 50)
        .attr('font-size', 12)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ED7D32')
        .attr('font-weight', 500)
        .text('Fair');
    divSvg.append('line')
        .attr('x1', XScale(fairValue)).attr('y1', 50+5)
        .attr('x2', XScale(fairValue)).attr('y2', svg_height-fairMetricDim.margin.bottom)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3 3') 
        .attr('stroke', '#ED7D32');

    // visualize each part
    divSvg.selectAll('metric').data(metricData).enter().append('g')
        .each(function(d, i){
            let yCenter = fairMetricDim.margin.top + barLen*i+barLen/2;
            let key = Object.keys(d)[0]

            let curColor = fontColor;
            if(i == metricData.length -1){
                curColor = '#5B9BD5';
            }

            // visualize the text
            d3.select(this).append('text')
                .attr('x', fairMetricDim.margin.left-gap/2).attr('y', yCenter)
                .attr('dy', '0.5em')
                .attr('text-anchor', 'end')
                .text(key)
                .attr('font-size', 12)
                .attr('font-weight', ()=>i == metricData.length-1 ? 600 : 'none')
                .attr('fill', curColor)
                .attr('stroke-width', 0)
                .classed('lastMetric',  i == metricData.length -1? true:false);

            // visualize separate line
            d3.select(this).append('line')
                .attr('x1', XScale(range[0])).attr('y1', yCenter+barLen/2)
                .attr('x2', XScale(range[1])).attr('y2', yCenter+barLen/2)
                .attr('stroke', 'grey')
                .attr('stroke-width', 0.2);
                

            // visualize the bars
            d3.select(this).append('line')
                .attr('x1', XScale(d[key])).attr('y1', yCenter-barLen/2+5)
                .attr('x2', XScale(d[key])).attr('y2', yCenter+barLen/2-5)
                .attr('stroke', curColor)
                .attr('stroke-width', '3px')
                .classed('lastMetric',  i == metricData.length -1? true:false);

            // visualize the value
            d3.select(this).append('text')
                .attr('x', ()=>{
                    if(d[key]<fairValue && XScale(d[key])-XScale(fairValue)<30){
                        return  XScale(d[key])-5;
                    }
                    else{
                        return  XScale(d[key])+5;
                    }
                })
                .attr('y', yCenter)
                .attr('dy', '0.5em')
                .attr('text-anchor', ()=>{
                    if(d[key]<fairValue && XScale(d[key])-XScale(fairValue)<30){
                        return 'end';
                    }
                    else{
                        return  'start';
                    }
                })
                .text(d[key])
                .attr('fill', curColor)
                .attr('fill-opacity', 0.6)
                .attr('font-size', 10)
                .attr('stroke-width', 0)
                .classed('lastMetric',  i == metricData.length -1? true:false);

        });
    
    // visualize the bias square
    divSvg.append('rect')
        .attr('x', XScale(range[0])).attr('y', svg_height-2*margin_y-fairMetricDim.margin.bottom + 23)
        .attr('width', 15).attr('height', 15)
        .attr('fill', baisAreaColor);
    
    // text
    divSvg.append('text').attr('x', XScale(range[0])+20)
        .attr('y', svg_height-2*margin_y-fairMetricDim.margin.bottom + 23)
        .attr('font-size', 10)
        .attr('dy', '1em')
        .attr('text-anchor', 'start')
        .text('Bias against woman')
        .attr('fill', fontColor);
}

/**
 * create an accuracy panel
 * @param {*} divSelector  the accuracy div
 * @param {*} accuracyData [{'original': 0.95}, {'mitigate': 0.2}...]
 */
function visAccuracyPanel(divSelector, accuracyData){
    // accuracyData = [{'baseline': 0.7}, {'mitigate': 0.8}];
    // accuracyData.push({'reweigh': 0.76})
    // accuracyData.push({'post': 0.68})

    // set the class
    divSelector.classed('accMetricDiv', true).classed('allCenter', true);

    // basic info
    let range = [0, 1];
    // let barLen = fairMetricDim.itemHei;
    let fontColor = '#213547';
    let gap = 10;       // the gap between two elements

    // init the div and the svg 
    let divWid = fairMetricDim.wid
    let divHei = fairMetricDim.margin.top + fairMetricDim.margin.bottom 
        + fairMetricDim.itemHei*accuracyData.length;
    divSelector
        .style('width', divWid+'px')
        .style('height', divHei+'px');
    let margin = 10;
    let svg_width = divWid - 2*margin;
    let svg_height = divHei - 2*margin
    let divSvg = divSelector.append('svg')
        .attr('width', svg_width)
        .attr('height', svg_height);

    let margin_bottom = margin;
    let innerWid = svg_width - fairMetricDim.margin.left - fairMetricDim.margin.right;
    let itemHei = (svg_height - fairMetricDim.margin.top - margin_bottom)/accuracyData.length;
    let barLen = itemHei;

    // visualization part
    let XScale = d3.scaleLinear()       // the scale for the metric value 
        .domain(range)
        .range([fairMetricDim.margin.left, fairMetricDim.margin.left+innerWid]);
    
     // viusualize the axis
     let tickSize = 5;
     let xAxis = d3.axisBottom(XScale).ticks(5).tickSize(tickSize);
     let axisG = divSvg.append('g')
         .attr('transform', `translate(0, ${fairMetricDim.margin.top})`)
         .call(xAxis);
     axisG.selectAll('line').attr('y2', -tickSize);      // reverse the ticks
     axisG.selectAll('text').attr('y', -15);         // change the y text
     axisG.select('path').remove();      // remove the previous one
     axisG.selectAll('line').attr('stroke', fontColor);
     axisG.selectAll('text').attr('fill', fontColor);
     axisG.append('line')
         .attr('x1', fairMetricDim.margin.left).attr('y1', 0)
         .attr('x2', fairMetricDim.margin.left+innerWid).attr('y2', 0)
         .attr('stroke', fontColor);
    
     // visualize the title
     divSvg.append('text').attr('x', svg_width/2+margin).attr('y', 25)
        .attr('font-size', 15)
        .attr('font-weight', 500)
        .attr('fill', fontColor)
        .attr('text-anchor', 'middle')
        .text('Accuracy');
    
    // visualize each part
    divSvg.selectAll('accuracy').data(accuracyData).enter().append('g')
        .each(function(d, i){
            let yCenter = fairMetricDim.margin.top + barLen*i+barLen/2;
            let key = Object.keys(d)[0]

            let curColor = fontColor;
            if(i == accuracyData.length -1){
                curColor = '#5B9BD5';
            }

            // visualize the text
            d3.select(this).append('text')
                .attr('x', fairMetricDim.margin.left-gap/2).attr('y', yCenter)
                .attr('dy', '0.5em')
                .attr('text-anchor', 'end')
                .text(key)
                .attr('font-size', 12)
                .attr('font-weight', ()=>i == accuracyData.length-1 ? 600 : 'none')
                .attr('fill', curColor);
            
            // visualize a bar
            d3.select(this).append('rect')
                .attr('x', XScale(range[0])).attr('y', yCenter-barLen/2+5)
                .attr('width', XScale(d[key])-XScale(range[0])).attr('height', barLen-10)
                .attr('fill', curColor)
                .attr('fill-opacity', 0.2);

            // visualize separate line
            d3.select(this).append('line')
                .attr('x1', XScale(range[0])).attr('y1', yCenter+barLen/2)
                .attr('x2', XScale(range[1])).attr('y2', yCenter+barLen/2)
                .attr('stroke', 'grey')
                .attr('stroke-width', 0.2);
                
            // visualize the bars
            d3.select(this).append('line')
                .attr('x1', XScale(d[key])).attr('y1', yCenter-barLen/2+5)
                .attr('x2', XScale(d[key])).attr('y2', yCenter+barLen/2-5)
                .attr('stroke', curColor)
                .attr('stroke-width', '3px');

            // visualize the value
            d3.select(this).append('text')
                .attr('x', XScale(d[key])+5).attr('y', yCenter)
                .attr('dy', '0.5em')
                .attr('text-anchor', 'begin')
                .text(d[key])
                .attr('fill', curColor)
                .attr('fill-opacity', 0.6)
                .attr('font-size', 10);

        });
    
    

}

/**
 * visualize the train or test panel as bar chart
 * @param {*} divSelector  the selector
 * @param {*} data  [[male_positive, male_negative], [female_positive, female_negative]]
 * @param {*} title  'Train Data'/'Test Data'
 * @param {*} weights weights for [[male_positive, male_negative], [female_positive, female_negative]]
 */
function visTrainOrTestData(divSelector, data, title, weights=undefined){
    let hasWeight = weights? true:false;
    let Rdata = hasWeight? [[data[0][0]*weights[0][0], data[0][1]*weights[0][1]], [data[1][0]*weights[1][0], data[1][1]*weights[1][1]]]:data;

    let div_width = parseInt(divSelector.style("width"));
    let div_height = parseInt(divSelector.style("height"));

    if(divSelector.attr("id")){
        let divSelector_dom = document.getElementById(divSelector.attr("id"));
        div_width = divSelector_dom.clientWidth;
        div_height = divSelector_dom.clientHeight;
    }

    let margin = 20;
    let svg_width = div_width;
    let svg_height = div_height - margin;
    let bar_width = (svg_width-margin) / (data.length+1) / 2
    
    // the X & Y axis scale for the bar chars
    let maxY = 0;
    Rdata.forEach(d=>{
        let maxD = Math.max(...d);
        if(maxD > maxY){
            maxY = maxD;
        }
    })
    
    let yScaleBar = d3.scaleLinear()
        .domain([0, maxY])
        .range([margin, svg_height-2*margin]);   
    let xScaleBar = (i)=> bar_width/2 + bar_width * (3*i);

    divSelector.selectAll('*').remove();
    divSelector.classed('dataContainer', false);
    
    // add a svg and change the size of the svg
    let svg = divSelector.append('svg')
        .attr("width", svg_width)
        .attr("height", svg_height);
    
    let svg_g = svg.append("g")
        .attr("transform", `translate(${margin/2}, 0)`)

    // visualize the bars
    let barG = svg_g.append('g')
    barG.selectAll('g').data(data)
        .enter().append('g')
        .each(function(d, i){
            // visualize two bars and add texts
            let xCenter = xScaleBar(i);
            let visbar = (num, x, label, weight)=>{
                d3.select(this).append('rect').attr('x', x).attr('y', svg_height-margin-yScaleBar(num*weight))
                    .attr('width', bar_width)
                    .attr('height', yScaleBar(num*weight))
                    .attr('fill', label==1? colorMap.orange : colorMap.blue)
                    .attr('border', 'none');
                // add text label for each bar
                if(hasWeight){
                    let addText = (pNum, anchor)=>{
                        return d3.select(this).append('text').attr('x', x+bar_width/2).attr('y', svg_height-margin-yScaleBar(num*weight))
                                .attr('text-anchor', anchor)
                                .attr('dy', '-0.2em')
                                .attr('class', 'fontFamily barNum')
                                .text(pNum);
                    }
                    addText(num, 'end');
                    addText(`x${weight}`, 'start').style('font-weight', 600).style('font-size', '8px').style('fill', '#E06666');
                }
                else{
                    d3.select(this).append('text').attr('x', x+bar_width/2).attr('y', svg_height-margin-yScaleBar(num*weight))
                        .attr('text-anchor', 'middle')
                        .attr('dy', '-0.2em')
                        .attr('class', 'fontFamily barNum')
                        .text(num);
                }
            }
            if(hasWeight){
                visbar(d[0], xCenter, 1, weights[i][0]);
                visbar(d[1], xCenter+bar_width, 0, weights[i][1]);
            }
            else{
                visbar(d[0], xCenter, 1, 1);
                visbar(d[1], xCenter+bar_width, 0, 1);
            }
            
            // viualize the x-axis label
            d3.select(this).append('text').attr('x', xCenter+bar_width/2).attr('y', yScaleBar(maxY)+margin)
                // .attr('text-anchor', 'middle')
                .attr('class', 'axisTick')
                .attr('dy', '1.2em')
                .text(attrVs[i])
                .attr('class', 'fontFamily ticks');
            
        })
    
    barG.selectAll('text').raise();
    // visualize the peripherals 
    // visualize the x axis
    svg_g.append('line') 
        .attr('x1', 5)
        .attr('y1', margin+yScaleBar(maxY))
        .attr('x2', svg_width-5-margin)
        .attr('y2', margin+yScaleBar(maxY))
        .attr('stroke', '#44546A');
    
    // visualize the title of this panel
    divSelector.append('span').classed('plotTitle', true).text(title)
        .style("top", `${svg_height}px`)
        .style("height", "20px");
    return divSelector;
}

/**
 * render a line on svg
 * @param {*} svg 
 * @param {*} startPoint [x, y]
 * @param {*} endPoint [x, y] markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
 */
function renderArrow(svg, startPoint, endPoint){
    // define the arrow
    if(svg.select('#arw').empty()){
        svg.append('defs').append('marker')
            .attr('id', 'arw')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('refX', 6)
            .attr('refY', 3)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M 0 0 L 6 3 L 0 6')
            .attr('stroke', 'none')
            .attr('fill', '#8497B0');
    }
    
    let pointLst = [startPoint, endPoint];
    if(startPoint[1]!=endPoint[1]){
        let middlePoint = [endPoint[0], startPoint[1]];
        pointLst = [startPoint, middlePoint, endPoint];
    }

    // add the row
    let arrowSelector = svg.append('path')
        .attr('d', d3.line()(pointLst))
        .attr('stroke', '#8497B0')
        .attr('marker-end', 'url(#arw)')
        .attr('fill', 'none');

    return arrowSelector;
}



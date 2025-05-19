/**
 * This file is used to visualize the fairMetrics of the original train and test data for the final project
 */

class InputFairMetricsFPanel{
    constructor() {
        this.containerSelector = d3.select('#contentDiv').append('div')
            .classed('visualComponentContainer', true)
            .classed('newVisualComponent', true);
        this.containerSelector.style('margin', '30px auto 30px auto;').attr('overflow', 'hidden');
        this.containerSelector.style('display', 'block').style('height', '110px').classed("placeHolder", true);
        // this.placeholder = this.containerSelector.append('div').classed('placeholder', true);
        this.dim = {
            dataWid: 100,    // the width of train name
            modelWid: 100,   // the width of model name
            metricsWid: 170,    // the width of each metric
            titleHei: 30,   // the height of first row
            axisHei: 20,    // the height of the second row
            rowHei: 30,      // the height of each row
            metricNum: 0,     // the number of metrics 
            rowNum: 0   // the number of rows
        }
        this.metricNames =['SPD', 'DI'];
        this.activate();
    }

    /**
     * fetch the accuracy data or the fairness metric data from the backend, and then render it on the page
     */
    activate(){
        // 1. remove all content in this div
        this.containerSelector.style('display', 'block').selectAll('*').remove();
        this.containerSelector.style('margin', '30px auto 30px auto;');
        this.containerSelector.style('display', 'block').style('height', '150px').classed("placeHolder", false);
               
        let data = [{'dataName': 'Training Data', 'SPD': -0.21, 'DI': 0.52}]
        // 3. update the dim and resize the grid div
        this.dim.metricNum =2;
        this.dim.rowNum = data.length;
        let gridWid = this.dim.dataWid+this.dim.metricsWid*this.dim.metricNum+this.dim.modelWid;
        let gridHei = this.dim.titleHei+this.dim.axisHei+(this.dim.rowNum+1)*this.dim.rowHei;
        this.gridSelector = this.containerSelector.append('div').classed('faiMetricsFDiv', true)
            .classed('allCenter', true);
        this.gridSelector.style('width', gridWid+'px').style('height', gridHei+'px');

        // 4. update the number of row and column 
        this.gridSelector.style('grid-template-columns', 
            `${this.dim.dataWid+this.dim.modelWid}px repeat(${this.dim.metricNum}, ${this.dim.metricsWid}px)`)
            .style('grid-template-rows',
            `${this.dim.titleHei}px ${this.dim.axisHei}px repeat(${this.dim.rowNum}, ${this.dim.rowHei}px)`);
        
        // update the height
        this.containerSelector.style('height', 150+this.dim.rowNum*30+'px');

        // 5. render the first two rows according
        this.renderTitleRows();
    
        // 6. render other rows
        this.renderRows(data);

        // 7. render the legend
        this.renderLegend();

        // hidden all of the scroll bar
        this.containerSelector.selectAll('div').style('overflow', 'hidden');
    }

    /**
     * render the first two rows
     */
    renderTitleRows(){
        // the Train Data and Model
        this.gridSelector.append('div').style('grid-row', '1/span 2').style('grid-column', '1/span 1')
            .classed('titleBBor', true).classed('title', true).classed('rBor', true).classed('metricsBg', false)
            .text('Data');
        // the metric names
        for(let i = 0; i < this.dim.metricNum; i++){
            this.gridSelector.append('div').classed('metricTitle', true).classed('rBor', true).classed('metricsBg', true)
                .text(fairMetricInfo[this.metricNames[i]].fullName);
        }
        // the four metric axis
        for(let i = 0; i < this.dim.metricNum; i++){
            let divSelector = this.gridSelector.append('div').classed('rBor', true)
                .classed('metricsBg', true);
            this.renderAxis(divSelector, this.metricNames[i]);
        }
    }

    // render axis for the fair metrics or the accuracy
    // type: acc , or fair metric names
    renderAxis(divSelector, type){
        let divWid = this.dim.metricsWid;
        let divHei = this.dim.axisHei;
        console.log('render axis:', divWid);
        let gap = 10;
        let range = type == 'acc'? [0, 1]:fairMetricInfo[type].range;
        // the scale
        let XScale = d3.scaleLinear()
            .domain(range)
            .range([gap, divWid-gap]);
        // color
        let fontColor = '#213547';
        
        // creat a svg
        let divSvg = divSelector.append('svg').style('width', divWid+'px')
            .style('height', divHei+'px');
        // add a axis on this svg
        let tickSize = 5;
        let xAxis = d3.axisBottom(XScale).ticks(5).tickSize(tickSize);
        let axisG = divSvg.append('g')
            .attr('transform', `translate(0, ${divHei})`)
            .call(xAxis);
        axisG.selectAll('line').attr('y2', -tickSize);      // reverse the ticks
        axisG.selectAll('text').attr('y', -15);         // change the y text
        axisG.select('path').remove();      // remove the previous one
        axisG.selectAll('line').attr('stroke', fontColor);
        axisG.selectAll('text').attr('fill', fontColor);
        axisG.append('line')
            .attr('x1', 0).attr('y1', 0)
            .attr('x2', divWid).attr('y2', 0)
            .attr('stroke', fontColor);
    }

    /**
     * render the general rows
     * @param {*} rowData 
     *  [{'trainName': , 'modelName': , 'SPD': , 'DI': , 'EOD':, 'AOD':}, ...]
     */
    renderRows(rowData){
        // first render the trainName and model Name
        rowData.forEach(ele => {
            this.gridSelector.append('div').classed('rBor', true).classed('rowText', true).classed('btmBor', true)
                .text(ele['dataName']);
            // render each metric
            for(let j = 0; j < this.metricNames.length; j++){
                let metricName = this.metricNames[j];
                let divSelector = this.gridSelector.append('div').classed('rBor', true).classed('btmBor', true);
                this.renderMetricVlaues(divSelector, metricName, ele[metricName]);
            }
        });
    }

    // visualize the metrics vlaues using visualization
    // type: acc, SPD, ...
    renderMetricVlaues(divSelector, type, value){
        // find the xscale
        let divWid = this.dim.metricsWid;
        let divHei = this.dim.rowHei;
        let gap = 10;
        let range = type == 'acc'? [0, 1]:fairMetricInfo[type].range;
        // the scale
        let XScale = d3.scaleLinear()
            .domain(range)
            .range([gap, divWid-gap]);
        // the x value of fairness line
        let fairValue = type == 'acc'? '':fairMetricInfo[type].fair;
        let baisAreaColor = '#EDEDED';  
        let fontColor = '#5B9BD5';  // #5B9BD5 #213547
        // bar length
        let barLen = divHei-gap;

        // create a svg
        let divSvg = divSelector.append('svg').style('width', divWid+'px')
            .style('height', divHei+'px');

        //visualize the fairness line
        divSvg.append('line')
            .attr('x1', XScale(fairValue)).attr('y1', 0)
            .attr('x2', XScale(fairValue)).attr('y2', divHei)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '3 3') 
            .attr('stroke', '#ED7D32');
        // visualize the baised area
        divSvg.append('rect')
            .attr('x', 0).attr('y', 0)
            .attr('width', XScale(fairValue)).attr('height', divHei)
            .attr('fill', baisAreaColor)
            .attr('fill-opacity', '0.5');
        
        // vsualize the bars
        divSvg.append('line')
            .attr('x1', XScale(value)).attr('y1', divHei/2-barLen/2)
            .attr('x2', XScale(value)).attr('y2', divHei/2+barLen/2)
            .attr('stroke', fontColor)
            .attr('stroke-width', '3px');
        // visualize the text
        divSvg.append('text')
            .attr('x', ()=>{
                if(value<fairValue && XScale(value)-XScale(fairValue)<30){
                    return  XScale(value)-5;
                }
                else{
                    return  XScale(value)+5;
                }
            })
            .attr('y', divHei/2)
            .attr('dy', '0.5em')
            .attr('text-anchor', ()=>{
                if(value<fairValue && XScale(value)-XScale(fairValue)<30){
                    return 'end';
                }
                else{
                    return  'start';
                }
            })
            .text(value)
            .attr('fill', fontColor)
            .attr('fill-opacity', 0.6)
            .attr('font-size', 10)
            .attr('stroke-width', 0);
    }

    // render legends 
    renderLegend(){
        let baisAreaColor = '#EDEDED';  
        let fontColor = '#213547';
        let emptyNum =1;
        for(let i=0; i<emptyNum; i++){
            this.gridSelector.append('div');
        }

        // render the fair legend
        let fairDiv = this.gridSelector.append('div');
        // let divHei = parseInt(fairDiv.style('height'));
        let divHei = this.dim.rowHei;
        let fairDivSvg = fairDiv.append('svg').style('width', fairDiv.style('width'))
            .style('height', divHei + 'px');
        fairDivSvg.append('line')
            .attr('x1', 0).attr('y1', divHei/2)
            .attr('x2', 30).attr('y2', divHei/2)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '3 3') 
            .attr('stroke', '#ED7D32');
        // text
        fairDivSvg.append('text').attr('x', 40)
            .attr('y', divHei/2)
            .attr('font-size', 10)
            .attr('dy', '0.5em')
            .attr('text-anchor', 'start')
            .text('Fair')
            .attr('fill', fontColor);

        // visualize the bias area
        let biasDiv = this.gridSelector.append('div');
        // divHei = parseInt(biasDiv.style('height'));
        divHei = this.dim.rowHei; 
        let biasDivSvg = biasDiv.append('svg').style('width', biasDiv.style('width'))
            .style('height', divHei + 'px');
        let rectH = 15;
        biasDivSvg.append('rect')
            .attr('x', 0).attr('y', (divHei-rectH)/2)
            .attr('width', rectH).attr('height', rectH)
            .attr('fill', baisAreaColor);
        // text
        biasDivSvg.append('text').attr('x', 30)
            .attr('y', divHei/2)
            .attr('font-size', 10)
            .attr('dy', '0.5em')
            .attr('text-anchor', 'start')
            .text('Biased')
            .attr('fill', fontColor);
    }

}

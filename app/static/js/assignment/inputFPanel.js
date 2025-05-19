/**
 * input panel for the project panel
*/
class InputFPanel{
    constructor(templateId, VCSet){
        this.name = 'final-input';
        // clone the div
        const visualComponentNode = document.getElementById(templateId);
        const cloneNode = visualComponentNode.content.cloneNode(true).querySelector('.visualComponentContainer');
        document.getElementById('contentDiv').appendChild(cloneNode);         // add the clone div into the webpage
        d3.select(cloneNode).classed('newVisualComponent', true);
        // panels
        this.inputDiv = d3.select('.assignment-input');    // the entire div
        this.tainPanel = this.inputDiv.select('.trainDataDiv');
        this.testPanel = this.inputDiv.select('.testDataDiv');
        this.legendPanel = this.inputDiv.select('.legendDiv');
        this.VCSet = VCSet;

        this.activate();
    }

    async activate(){
        await axios.post('/startProject')
            .then((response)=>{ // getTrainTest
                // change the datatype as bank
                outputLabel = outputDict['Bank'];
                attrVs = attrValueDict['Bank'];
                yScaleBar = yScaleBarDict['Bank'];
                let data = response.data;
                this.trainData = data['train'];
                this.testData = data['test'];
                this.VCSet.data.trainData.Original = this.trainData;
                this.VCSet.data.testData = this.testData;
                styleLegend(this.legendPanel);
                visTrainOrTestData(this.tainPanel, this.trainData, 'Training Data', undefined, ['Married', 'Unmarried']);
                visTrainOrTestData(this.testPanel, this.testData, 'Test Data', undefined, ['Married', 'Unmarried']);
            })
            .catch((error)=>{
                console.log(error);
            });
    }
    
}
/**
 * This class is used to generate the customization panel about the train data and model
 */
class TrainModelCusPanel{
    constructor(setPosition, VCSet) {
        this.name = 'trainModelCus';
        this.modelMap = {'Logistic Regression': 'LR', 'Prejudice Remover': 'PrejudiceRmv', 'Adversarial Debiasing': 'Adversarial'};
        this.trainMap = {'Original': 'Original', 'Reweighing': 'Reweighing', 'Learning Fair Representations': 'LFR', 
            'Optimized Preprocessing': 'OptimPreproc'};
        this.trainName = 'Original';
        this.model = 'LR';
        this.setPosition = setPosition;
        this.VCSet = VCSet;

        this.containerSelector = d3.select('#contentDiv').append('div')
            .classed('visualComponentContainer', true)
            .classed('newVisualComponent', true)
            .style('display', 'block').style('height', '600px'); 

        this.containerId = this.containerSelector.attr("id")
        // add container panel
        this.cusTMSelector = this.containerSelector.append('div').classed('cusTMContrainer', true);
        // title panel
        this.titleSelector = this.cusTMSelector.append('div').classed('childDiv', true).classed('attrTitle', true)
            .text('Train Data & Model Customization');
        // train data selection
        this.trainSelector = this.cusTMSelector.append('div').classed('childDiv', true);
        this.trainSelector.append('span').classed('labelClass', true).text('Train Data');
        // pre-processing div
        this.preProSelector = this.cusTMSelector.append('div').classed('childDiv', true);
        // model selection
        this.modelSelector = this.cusTMSelector.append('div').classed('childDiv', true);
        this.modelSelector.append('span').classed('labelClass', true).text('Model');
        // button div
        this.btnDivSelector = this.cusTMSelector.append('div').classed('childDiv', true);
        this.submitBtn = '';
        this.addTrainBtns();
        this.addModelBtns();
        this.addSubmitBtn();
        this.renderPreprocessing('Original');
    }

    addTrainBtns(){
        let that = this;
        let trainName = ['Original', 'Reweighing', 'Learning Fair Representations', 'Optimized Preprocessing'];
        this.trainSelector.selectAll('btn').data(trainName).enter()
            .append('button')
            .attr('type', 'button')
            .classed('featureBtn', true)
            .classed('trainBtns', true)
            .style('background', (d)=>{
                return d == 'Original'? '#B3C7E7' : null;
            })
            .style('border-color', (d)=>{
                return d == 'Original'? '#4372C4' : null;
            })
            .style('color', (d)=>{
                return d == 'Original'? '#4372C4' : null;
            })
            .text(d=>d)
            .on('click', function(_, d){
                if(d != 'Original'){
                    that.trainName = that.trainMap[d];;
                    that.trainNameCandi = d;
                }
                else{
                    that.trainName = that.trainMap[d];
                }
                // all other restore
                that.trainSelector.selectAll('.trainBtns')
                    .style('background', '#EDEDED')
                    .style('border-color', '#C9C9C9')
                    .style('color', null)
                d3.select(this).style('background', '#B3C7E7')
                    .style('border-color', '#4372C4')
                    .style('color', '#4372C4');
                // render the new
                that.renderPreprocessing(d);
            });
    }

    addModelBtns(){
        let that = this;
        let ModelName = ['Logistic Regression', 'Prejudice Remover', 'Adversarial Debiasing'];
        this.modelSelector.selectAll('btn').data(ModelName).enter()
            .append('button')
            .attr('type', 'button')
            .classed('featureBtn', true)
            .classed('modelBtns', true)
            .style('background', (d)=>{
                return d == 'Logistic Regression'? '#B3C7E7' : null;
            })
            .style('border-color', (d)=>{
                return d == 'Logistic Regression'? '#4372C4' : null;
            })
            .style('color', (d)=>{
                return d == 'Logistic Regression'? '#4372C4' : null;
            })
            .text(d=>d)
            .on('click', function(_, d){
                // all other restore
                that.model = that.modelMap[d];
                that.modelSelector.selectAll('.modelBtns')
                    .style('background', '#EDEDED')
                    .style('border-color', '#C9C9C9')
                    .style('color', null);
                d3.select(this).style('background', '#B3C7E7')
                    .style('border-color', '#4372C4')
                    .style('color', '#4372C4');
            });
    }

    addSubmitBtn(){
        this.submitBtn = this.btnDivSelector.append('button').classed('submitBtn', true).text('Submit')
            .on('click', ()=>{
                this.submit();
            });
    }

    async renderPreprocessing(type){
        if(type == 'Original'){
            this.preProSelector.selectAll('*').remove();
            let trainSelector = this.preProSelector.append('div')
                .attr('class', 'trainDataDiv trainTestDDiv');
                // .style("width", "200px")
                // .style("height", "300px");            
            visTrainOrTestData(trainSelector, this.VCSet.data.trainData.Original, 'Train Data', undefined, ['Married', 'Unmarried']);
            // this.preProSelector.style('height', vwTopx(parseInt(trainSelector.style('height')))+50+'px');
            this.preProSelector.style('height', parseInt(trainSelector.style('height'))+50+'px');
            trainSelector.classed('allCenter', true);
            // resize the biggest container
            this.containerSelector.style('height', (this.cusTMSelector.style('height'))+50+'px');
        }
        else{
            this.preProSelector.selectAll('*').remove();
            let trainSelector = this.preProSelector.append('div')
                .attr('class', 'trainDataDiv trainTestDDiv');
            let typeAbbre = this.trainMap[type];

            await axios.post('/getPreprocessData', {'type': this.trainMap[type]})
                .then((response)=>{
                let data = response.data;
                // if in a set then save the processed train data into the VCset
                if(typeAbbre == 'Reweighing'){
                    this.deBiasedTrainData = data;
                    if(this.VCSet){this.VCSet.data.trainData[typeAbbre] = this.deBiasedTrainData;} // we only store the weights of groups
                    visTrainOrTestData(trainSelector, this.VCSet.data.trainData.Original, 'Training Data', this.deBiasedTrainData, ['Married', 'Unmarried']);
                }
                else if(typeAbbre == 'LFR'){
                    this.deBiasedTrainData = data['train'];
                    if(this.VCSet){
                        this.VCSet.data.trainData[typeAbbre] = this.deBiasedTrainData;
                        this.VCSet.data.testData = data['test'];
                    }
                    visTrainOrTestData(trainSelector, this.deBiasedTrainData, 'Training Data after '+this.type, '', ['Married', 'Unmarried']);
                }
                else if(typeAbbre == 'OptimPreproc'){
                    this.deBiasedTrainData = data['train'];
                    if(this.VCSet){
                        this.VCSet.data.trainData[typeAbbre] = this.deBiasedTrainData;
                        this.VCSet.data.testData = data['test'];
                    }
                    visTrainOrTestData(trainSelector, this.deBiasedTrainData, 'Training Data after '+this.type, '', ['Married', 'Unmarried']);
                }
                // this.preProSelector.style('height', vwTopx(parseInt(trainSelector.style('height')))+50+'px');
                this.preProSelector.style('height', parseInt(trainSelector.style('height'))+50+'px');
                trainSelector.classed('allCenter', true);
                // resize the biggest container
                this.containerSelector.style('height', (this.cusTMSelector.style('height'))+50+'px');
            })
            .catch((error)=>{
                console.log(error);
            });
            // let preProcessor = new PreProcessPanel(this.preProSelectorContainer, this.trainMap[type], -2, this.VCSet);
            // preProcessor.trainData = this.VCSet.data.trainData.Original;
            // preProcessor.activate();
            // preProcessor.getProcessedData();
        }
    }

    /**the preprocess finished */
    preProDone(){
        // save the train name
        this.trainName = this.trainMap[this.trainNameCandi];
        // enable the submit button
        this.submitBtn.on('click', ()=>{
            this.submit();
        });
    }

    /**
     * submit button
     */
    submit(){
        if(this.model && this.trainName){
            let mlPipeline = this.VCSet.findVC('MLPipeline', this.setPosition);
            if(mlPipeline){
                mlPipeline.activate(this.trainName, this.model);
            }
        }
        else{
            // alert('Please Select a Train Data!');
        }
    }

    /**enable all buttons on this page */
    enableBtns(){
        if(this.submitBtn){
            this.submitBtn.on('click', ()=>this.submit());
        }
    }

    /**disable all buttons on this page */
    disableBtns(){
        if(this.submitBtn){
            this.submitBtn.on('click', null);
        }
    }
}
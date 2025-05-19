
class AccuracyPanel{
    /**
     * Accuracy panel
     * @param {*} containerSelector 
     * @param {*} type
     */
    constructor(containerSelector, type){
        this.type = type;
        this.containerSelector = containerSelector;
        this.containerSelector
            .style('margin', '30px auto 30px auto')
            .style('display', 'block')
            .classed("placeHolder", true);
    }
    /**
     * fetch the accuracy data from the backend, and then render it on the page
     */
    async activate(){
        this.reset();
        await axios.post('/getAccuracy', {
            type: this.type
        })
        .then((response)=>{
            this.accuData = response.data;
            console.log('Accuracy data', this.accuData);
        })
        .catch((error)=>{
            console.log(error);
        });
        this.containerSelector
            .style('height', '200px')
            .classed("placeHolder", false);

        // add the accuracyPanel
        let divSelector = this.containerSelector.append('div');
        visAccuracyPanel(divSelector, this.accuData);
        
        // update the table once again
        updateArrows();
    }

    /*restore to the palceholder*/
    reset(){
        this.containerSelector.selectAll('*').remove();
        this.containerSelector.style('margin', '30px auto 30px auto;')
        this.containerSelector.style('display', 'block').style('height', '110px');
    }
}


function triggerAccuracyPanel(containerSelector, type){
    return new AccuracyPanel(containerSelector, type);
}



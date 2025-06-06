/**
 * DESIGN NOTES
 * The enter script of this tool
 *      - validate the passcode
 *      - no need to validate again when refreshing the current page 
 *      - success validation leads to rendering the side nav and the first chapter
 */

let VALID = false;      
let initiation = '';      // the object of Initiation
let renderChapterObj = '';      // render the chapter handler

/**
 * validate password
 *      - if success => activate page; else repeat 5 times
 * @returns null
 */
async function validateWeb() {
    VALID = true;
    activateWeb();
    // // test if it's reload page
    // let reloadPage = sessionStorage.getItem('pageHasBeenLoaded');
    // if (reloadPage) {
    //     VALID = true;
    //     activateWeb();
    //     return '';
    // }

    // // if not, ask to enter password
    // let testNum = 1;
    // let pass = prompt('Please Enter Your Password',' ');
    // while (testNum < 5) {
    //     if (!pass) { history.go(-1) };
    //     let res = '';
    //     await axios.post('/verifyPwd', {pwd: pass.toLowerCase()})
    //         .then((response)=>{
    //             res = response.data['res'];
    //         })
    //         .catch((error)=>{
    //             console.log(error);
    //         });
    //     if (res == "yes") {
    //         VALID = true
    //         break;
    //     } 
    //     testNum += 1;
    //     pass = prompt('Access Denied - Password Incorrect, Please Try Again.','Password');
    // }
    // // test if successfully validate
    // if(VALID){
    //     activateWeb();
    //     sessionStorage.setItem('pageHasBeenLoaded', 'true');
    // }
    // else{
    //     history.go(-1);
    // }

    return '';
} 

/**
 * get the all chapter names and the first chapter's content, then render them
 */
function activateWeb(){
    axios.post('/init')
        .then((response)=>{
            let data = response.data;   // return title list and first chapter
            let chapterTitleLst = data['titles'];
            let chapterContent = data['firstChapter'];

            d3.select("body").style("visibility", "visible");
            renderChapterObj = new RenderChapterHandler();
            initiation = new Initiation(chapterTitleLst);
            isFinalProject = false;
            renderChapterObj.renderChapter(chapterContent, 1);     // render the first chapter content
        })
        .catch((error)=>{
            console.log(error);
        });
}



validateWeb();
const express = require("express");
const http = require("http");
const bodyParser = require("body-parser");
const https = require("https");
const multer  = require('multer')
const { spawn }  = require('child_process');
var storage = multer.diskStorage(
    {
        destination: './public/uploads/wav/',
        filename: function ( req, file, cb ) {
            cb( null, "MIDAS_"+file.originalname+".wav");
        }
    }
);
var upload = multer( { storage: storage } );


const hostname = "localhost";
const port = 4321;

const app = express();

app.use(bodyParser.json())

app.use(express.static(__dirname+"/public/"));
app.route("/")
.all((req,res,next)=>{
    res.statusCode = 200;
    res.setHeader("Content-Type", "text/html");
    // console.log("yi");
    next();
})
.get((req,res,next)=>{
    console.log("hi");
    res.sendFile(__dirname+"/public/index.html")
    // next()
})
.post(upload.single("file"),(req,res,next)=>{
    if (req.file) {
        console.log('Uploaded: ', req.file);
        console.log(req.body.audioText)
        console.log(req.body.age)
        console.log(req.body.gender)
        console.log(req.body.country)
        var address= "/public/uploads/wav/"+req.file.filename;
        
        var addressTxt = "/public/uploads/txt/" + req.file.filename.slice(0,-3)+"txt";
        console.log(addressTxt)
        var change = spawn('python3', [__dirname+"/public/code/change.py",address]);
        var saveMeta = spawn('python3', [__dirname+"/public/code/saveMetaData.py",addressTxt,req.body.audioText,req.body.age,req.body.gender,req.body.country]);
        var predict = spawn('python3', [__dirname+"/public/code/trans.py",address]);
        predict.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
            res.json({
                status:"success",
                name: `${data}` ,
                random: "ok"
            })
        });
          
        predict.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });
          
        predict.on('close', (code) => {
            console.log(`child process exited with code ${code}`);
        });
      }
});



const server = http.createServer(app);
server.listen(port,hostname,()=>{
    console.log(`Server running at http://${hostname}:${port}/`);
});

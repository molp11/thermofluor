var data = source.data;
var filetext = 'Well ID,Sample name,Tm,Error\n';    
for (var i = 0; i < data['w'].length; i++) {
    var currRow = [data['w'][i],
                   data['n'][i],
                   data['t'][i],
                   data['e'][i].toString().concat('\n')];
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = file_name;
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
} else {
    var link = document.createElement("a");
    link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'));
}

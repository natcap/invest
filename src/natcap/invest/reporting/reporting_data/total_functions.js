$(document).ready(function(){
    //Set the one time total sum
    sum_constant_total();

    $('[name="cb"]').change(function() {
        //On a check box change event

        //Get the table where the check box was changed
        var $table = $(this).closest('table');
        //Find the table data that represents checkbox total and set to
        //0 instead of --
        $table.find('.checkTot').text("0");
        $table.find('[name="cb"]:checked').closest('tr').find('.rowDataSd').each(function() {
            var $td = $(this);
            //Get table data value for checkbox total
            var $sumColumn = $table.find('tr.checkTotal td:eq(' + $td.index() + ')');
            //Get the current total or 0 if not defined
            var currVal = parseFloat($sumColumn.text().split(',').join('')) || 0;
            //Update the total value
            var upVal = +currVal + parseFloat($td.text().split(',').join(''));
            //Set new total value
            $sumColumn.html(upVal);
            });

        });
});

function sum_constant_total() {
    //This function sets the constant total row for each column

    //For each table get and set totals
    $('table').each(function(){
        var totals_array = new Array();
        //Get all the table rows except for the total row itself
        var $dataRows = $(this).find("tr:not('.totalColumn')");
        //For each row build up array
        $dataRows.each(function() {
            $(this).find('.rowDataSd').each(function(i){
                totals_array[i] = 0;
            });
        });
        //build up the values for each column over the rows
        $dataRows.each(function() {
            $(this).find('.rowDataSd').each(function(i){
                totals_array[i]+=parseFloat($(this).text());
            });
        });
        //Set the total values in the total row for each column
        $(this).find("td.totalCol").each(function(i){
            $(this).html(totals_array[i]);
        });
    });
}
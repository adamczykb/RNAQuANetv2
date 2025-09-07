package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var aPython = "/home/adamczykb/rnaquanet/pipelines/rnaquanet/run.py"
var aMaxitURL = "172.17.0.1:8080"

type ScriptOut struct {
	GraphPath string `json:"graph_path"`
	CsvPath   string `json:"csv_path"`
}

func saveFileToTemp(file io.Reader) (string, error) {
	f, err := os.CreateTemp("/tmp", "*.pdb")
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = io.Copy(f, file)
	if err != nil {
		return "", err
	}

	return f.Name(), nil
}

func getPDB(w http.ResponseWriter, r *http.Request) {
	file, handler, err := r.FormFile("file")
	_, err = os.Stat(aPython)
	if os.IsNotExist(err) {
		fmt.Printf("%s | Python script not found: %s\n", time.Now().Format(time.DateTime), aPython)
		return
	}
	if strings.HasSuffix(handler.Filename, ".pdb") {
		fmt.Printf("%s | File is a PDB file\n", time.Now().Format(time.DateTime))
	} else {
		fmt.Printf("%s | File is not a PDB file\n", time.Now().Format(time.DateTime))
		io.WriteString(w, "File is not a PDB file\n")
		return
	}
	if err != nil {
		io.WriteString(w, err.Error())
		return
	}
	fileName, err := saveFileToTemp(file)

	if err != nil {
		io.WriteString(w, err.Error())
		return
	}
	fmt.Printf("%s | File saved to %s\n", time.Now().Format(time.DateTime), fileName)
	var rmsd string = "0.00"
	if r.FormValue("RMSD") != "" {
		rmsd = r.FormValue("RMSD")
		fmt.Printf("%s | RMSD: %s\n", time.Now().Format(time.DateTime), rmsd)
	}
	script := exec.Command("python3", aPython, fmt.Sprintf("-f %s", fileName), "-r", rmsd)
	out, err := script.CombinedOutput()
	if err != nil {
		fmt.Printf("%s", fmt.Sprintf("%s | Error %s %s", time.Now().Format(time.DateTime), err.Error(), out))
	}
	os.Remove(fileName)
	var jsonOutput ScriptOut
	json.Unmarshal(out, &jsonOutput)
	fmt.Printf("%s | Graph path: %s\n", time.Now().Format(time.DateTime), jsonOutput.GraphPath)
	http.ServeFile(w, r, jsonOutput.GraphPath)
}
func getCIF(w http.ResponseWriter, r *http.Request) {
	file, handler, err := r.FormFile("file")
	_, err = os.Stat(aPython)
	if os.IsNotExist(err) {
		fmt.Printf("%s | Python script not found: %s\n", time.Now().Format(time.DateTime), aPython)
		return
	}
	if strings.HasSuffix(handler.Filename, ".cif") {
		fmt.Printf("%s | File is a mmCIF file\n", time.Now().Format(time.DateTime))
	} else {
		fmt.Printf("%s | File is not a mmCIF file\n", time.Now().Format(time.DateTime))
		io.WriteString(w, "File is not a mmCIF file\n")
		return
	}
	if err != nil {
		io.WriteString(w, err.Error())
		return
	}
	fileName, err := saveFileToTemp(file)

	requestURL := fmt.Sprintf("http://%s", aMaxitURL)
	content, err := os.ReadFile(fileName)
	if err != nil {
		fmt.Printf("%s | Reading mmCIF file error\n", time.Now().Format(time.DateTime))
		return
	}

	request, err := http.NewRequest("POST", requestURL, bytes.NewReader(content))
	request.Header.Set("X-Filename", filepath.Base(fileName))
	request.Header.Set("Content-Type", "text/plain")

	if err != nil {
		fmt.Printf("%s | error during converting pdb to mmCif %s\n", time.Now().Format(time.DateTime), err)
		return
	}

	client := &http.Client{}
	response, err := client.Do(request)

	if err != nil {
		io.WriteString(w, err.Error())
		return
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		fmt.Printf("%s | error during converting pdb to mmCif %s\n", time.Now().Format(time.DateTime), err)
		return
	}
	respBody, err := io.ReadAll(response.Body)

	tempFile, err := os.CreateTemp("/tmp", "*.pdb")
	if err != nil {
		fmt.Printf("%s | Error creating temporary file: %s\n", time.Now().Format(time.DateTime), err)
		return
	}
	defer tempFile.Close()

	_, err = tempFile.Write(respBody)
	if err != nil {
		fmt.Printf("%s | Error writing to temporary file: %s\n", time.Now().Format(time.DateTime), err)
		return
	}

	fileName = tempFile.Name()

	fmt.Printf("%s | File saved to %s\n", time.Now().Format(time.DateTime), fileName)
	var rmsd string = "0.00"
	if r.FormValue("RMSD") != "" {
		rmsd = r.FormValue("RMSD")
		fmt.Printf("%s | RMSD: %s\n", time.Now().Format(time.DateTime), rmsd)
	}
	script := exec.Command("python3", aPython, fmt.Sprintf("-f %s", fileName), "-r", rmsd)
	out, err := script.CombinedOutput()
	if err != nil {
		fmt.Printf("%s", fmt.Sprintf("%s | Error %s %s", time.Now().Format(time.DateTime), err.Error(), out))
	}
	os.Remove(fileName)
	var jsonOutput ScriptOut
	json.Unmarshal(out, &jsonOutput)
	fmt.Printf("%s | Graph path: %s\n", time.Now().Format(time.DateTime), jsonOutput.GraphPath)
	http.ServeFile(w, r, jsonOutput.GraphPath)

}

func main() {
	paramPort := flag.Int("port", 3000, "port to listen on")
	paramPython := flag.String("python", "/rnaquanet/run.py", "path to python script")
	paramMaxitURL := flag.String("maxiturl", "172.17.0.1:8080", "maxit url")

	flag.Parse()
	http.HandleFunc("/pdb", getPDB)
	http.HandleFunc("/cif", getCIF)

	println("Starting server on port", *paramPort)
	println("Path to python script", *paramPython)
	aPython = *paramPython
	aMaxitURL = *paramMaxitURL
	err := http.ListenAndServe(fmt.Sprintf(":%d", *paramPort), nil)

	if errors.Is(err, http.ErrServerClosed) {
		fmt.Printf("server closed\n")
	} else if err != nil {
		fmt.Printf("error starting server: %s\n", err)
		os.Exit(1)
	}
}
